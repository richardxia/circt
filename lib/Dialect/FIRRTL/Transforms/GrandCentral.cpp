//===- GrandCentral.cpp - Ingest black box sources --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Implement SiFive's Grand Central transform.  Currently, this supports
// SystemVerilog Interface generation.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/PointerSumType.h"
#include "llvm/ADT/StringSwitch.h"
#include <variant>

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct Element {
  enum ElementKind {
    Error = -1,
    Bundle,
    Vector,
    Ground,
    String,
    Boolean,
    Integer,
    Double
  };

  const ElementKind kind;

  StringRef name;

  uint64_t storage;

  ElementKind getKind() const { return kind; }
};

struct BundleKind : public Element {
  BundleKind(StringRef name) : Element({ElementKind::Bundle, name, 0}) {}
  static bool classof(const Element *e) {
    return e->getKind() == ElementKind::Bundle;
  }
};

struct VectorKind : public Element {
  VectorKind(StringRef name, uint64_t depth)
      : Element({ElementKind::Vector, name, depth}) {}
  static bool classof(const Element *e) {
    return e->getKind() == ElementKind::Vector;
  }
  uint64_t getDepth() const { return storage; }
};

struct GroundKind : public Element {
public:
  GroundKind(StringRef name, uint64_t width)
      : Element({ElementKind::Ground, name, width}) {}
  static bool classof(const Element *e) {
    return e->getKind() == ElementKind::Ground;
  }
  uint64_t getWidth() const { return storage; }
};

struct StringKind : public Element {
  StringKind(StringRef name) : Element({ElementKind::String, name, 0}) {}
  static bool classof(const Element *e) {
    return e->getKind() == ElementKind::String;
  }
};

struct BooleanKind : public Element {
  BooleanKind(StringRef name) : Element({ElementKind::Boolean, name, 0}) {}
  static bool classof(const Element *e) {
    return e->getKind() == ElementKind::Boolean;
  }
};

struct IntegerKind : public Element {
  IntegerKind(StringRef name) : Element({ElementKind::Integer, name, 0}) {}
  static bool classof(const Element *e) {
    return e->getKind() == ElementKind::Integer;
  }
};

struct DoubleKind : public Element {
  DoubleKind(StringRef name) : Element({ElementKind::Double, name, 0}) {}
  static bool classof(const Element *e) {
    return e->getKind() == ElementKind::Double;
  }
};

namespace {

/// The data content of an AugmentedBundleType.
struct BundleType {
  StringAttr defName;
  ArrayAttr elements;
  IntegerAttr id;

  /// Optionally construct a BundleType from an Annotation, returning None if
  /// the Annotation doesn't match the expected format.
  static Optional<BundleType> fromAnno(Annotation annotation) {
    auto defName = annotation.getMember<StringAttr>("defName");
    auto elements = annotation.getMember<ArrayAttr>("elements");
    bool failed = false;
    if (!defName) {
      llvm::errs() << "invalid 'AugmentedBundleType': missing 'defName' in "
                   << annotation.getDict();
      failed = true;
    }
    if (!elements) {
      llvm::errs() << "invalid 'AugmentedBundleType': missing 'defName' in "
                   << annotation.getDict();
      failed = true;
    }
    if (failed)
      return None;

    return BundleType(
        {defName, elements, annotation.getMember<IntegerAttr>("id")});
  }

  /// Return true if this AugmentedBundleType (a SystemVerilog interface) that
  /// is a top-level interface.  Or: this returns false if this interface is
  /// instantitate by another interface.  This works because only the outermost
  /// AugmentedBundleType has an ID.  This invariant is established due to
  /// annotations scattering logic.
  bool isRoot() { return id != nullptr; }
};

/// The data content of an AugmentedFieldType.
struct FieldType {
  StringAttr clazz;
  StringAttr name;
  StringAttr description;

  /// Optionally construct a FieldType from an Attribute, returning None if the
  /// Attribute does not match the expected format.
  static Optional<FieldType> fromAttr(Attribute attribute) {
    auto dictionary = attribute.dyn_cast_or_null<DictionaryAttr>();
    if (!dictionary) {
      llvm::errs() << "non-dictionary attribute used to construct a FieldType: "
                   << attribute << "\n";
      return None;
    }

    auto clazz = dictionary.getAs<StringAttr>("class");
    auto name = dictionary.getAs<StringAttr>("name");
    if (!name)
      name = dictionary.getAs<StringAttr>("defName");
    if (!clazz || !name) {
      llvm::errs() << "field missing 'class' or 'name'/'defName' in: "
                   << dictionary << "\n";
      return None;
    }
    return FieldType(
        {clazz, name, dictionary.getAs<StringAttr>("description")});
  }
};

/// Stores the information content of an ExtractGrandCentralAnnotation.
struct ExtractionInfo {
  /// The directority where Grand Central generated collateral (modules,
  /// interfaces, etc.) will be written.
  StringRef directory;

  /// The name of the file where any binds will be written.  This will be placed
  /// in the same output area as normal compilation output, e.g., output
  /// Verilog.  This has no relation to the `directory` member.
  StringRef bindFilename;
};

struct CompanionInfo {
  StringRef name;

  FModuleOp companion;

  FModuleOp mapping;
};

/// Remove Grand Central Annotations associated with SystemVerilog interfaces
/// that should emitted.  This pass works in three major phases:
///
/// 1. The circuit's annotations are examnined to figure out _what_ interfaces
///    there are.  This includes information about the name of the interface
///    ("defName") and each of the elements (sv::InterfaceSignalOp) that make up
///    the interface.  However, no information about the _type_ of the elements
///    is known.
///
/// 2. With this, information, walk through the circuit to find scattered
///    information about the types of the interface elements.  Annotations are
///    scattered during FIRRTL parsing to attach all the annotations associated
///    with elements on the right components.
///
/// 3. Add interface ops and populate the elements.
///
/// Grand Central supports three "normal" element types and four "weird" element
/// types.  The normal ones are ground types (SystemVerilog logic), vector types
/// (SystemVerilog unpacked arrays), and nested interface types (another
/// SystemVerilog interface).  The Chisel API provides "weird" elements that
/// include: Boolean, Integer, String, and Double.  The SFC implementation
/// currently drops these, but this pass emits them as commented out strings.
struct GrandCentralPass : public GrandCentralBase<GrandCentralPass> {
  void runOnOperation() override;

private:
  // Mapping of ID to leaf graound type associated with that ID.
  llvm::DenseMap<Attribute, Value> leafMap;

  // Mapping of ID to parent instance and module.
  llvm::DenseMap<Attribute, std::pair<InstanceOp, FModuleOp>> parentIDMap;

  // Mapping of ID to companion module.
  llvm::DenseMap<Attribute, CompanionInfo> companionIDMap;

  bool traverseField(Attribute maybeField, SmallVector<Element> &tpe,
                     IntegerAttr id, Twine path, bool buildIFace = false);

  bool traverseBundle(Annotation anno, IntegerAttr id = {}, Twine path = {},
                      bool buildIFace = false);

  void buildInterfaceSignal(ArrayRef<Element> elements, OpBuilder &builder,
                            StringRef name, StringAttr description);

  FModuleOp getEnclosingModule(Value value);

  // Inforamtion about how the circuit should be extracted.  This will be
  // non-empty if an extraction annotation is found.
  llvm::Optional<ExtractionInfo> maybeExtractInfo = None;

  StringRef getOutputDirectory() {
    return maybeExtractInfo ? maybeExtractInfo.getValue().directory : ".";
  }

  InstancePaths *instancePathsRef;
};
} // namespace

bool GrandCentralPass::traverseField(Attribute maybeField,
                                     SmallVector<Element> &tpe, IntegerAttr id,
                                     Twine path, bool buildIFace) {
  auto field = maybeField.dyn_cast_or_null<DictionaryAttr>();
  if (!field) {
    llvm::errs() << "non-dictionary field: " << maybeField << "\n";
    return false;
  }
  auto clazz = field.getAs<StringAttr>("class");
  if (!clazz) {
    llvm::errs() << "missing 'class' in field: " << field << "\n";
    return false;
  }

  auto name = field.getAs<StringAttr>("name");
  auto defName = field.getAs<StringAttr>("defName");
  switch (llvm::StringSwitch<Element::ElementKind>(clazz.getValue())
              .Case("sifive.enterprise.grandcentral.AugmentedBundleType",
                    Element::Bundle)
              .Case("sifive.enterprise.grandcentral.AugmentedVectorType",
                    Element::Vector)
              .Case("sifive.enterprise.grandcentral.AugmentedGroundType",
                    Element::Ground)
              .Case("sifive.enterprise.grandcentral.AugmentedStringType",
                    Element::String)
              .Case("sifive.enterprise.grandcentral.AugmentedBooleanType",
                    Element::Boolean)
              .Case("sifive.enterprise.grandcentral.AugmentedIntegerType",
                    Element::Integer)
              .Case("sifive.enterprise.grandcentral.AugmentedDoubleType",
                    Element::Double)
              .Default(Element::Error)) {
  case Element::Bundle: {
    if (!defName) {
      llvm::errs() << "missing 'defName' in BundleType: " << field << "\n";
      return false;
    }
    if (buildIFace)
      tpe.push_back(std::move(BundleKind(defName.getValue())));
    return traverseBundle(Annotation(field), id, path, buildIFace);
  }
  case Element::Vector: {
    auto elements = field.getAs<ArrayAttr>("elements");
    if (!name || !elements) {
      llvm::errs() << "missing 'name' or 'elements' in VectorType: " << field
                   << "\n";
      return false;
    }
    if (buildIFace)
      tpe.push_back(
          std::move(VectorKind(name.getValue(), elements.getValue().size())));
    bool notFailed = true;
    for (size_t i = 0, e = elements.size(); i != e; ++i) {
      notFailed &=
          traverseField(elements[i], tpe, id, path + "[" + Twine(i) + "]",
                        (i == 0) && buildIFace);
    }
    return notFailed;
  }
  case Element::Ground: {
    auto groundID = field.getAs<IntegerAttr>("id");
    auto leafValue = leafMap.lookup(groundID);
    if (!groundID || !leafValue) {
      llvm::errs() << "missing 'id' or 'leafValue' in GroundType: " << field
                   << "\n";
      return false;
    }

    auto builder =
        OpBuilder::atBlockEnd(companionIDMap.lookup(id).mapping.getBodyBlock());

    auto srcPaths =
        instancePathsRef->getAbsolutePaths(getEnclosingModule(leafValue));
    assert(srcPaths.size() == 1 &&
           "Unable to handle multiply instantiated companions");
    SmallString<0> srcRef;
    for (auto path : srcPaths[0])
      srcRef.append((path.name() + ".").str());

    auto uloc = builder.getUnknownLoc();
    if (auto blockArg = leafValue.dyn_cast<BlockArgument>()) {
      FModuleOp module = cast<FModuleOp>(blockArg.getOwner()->getParentOp());
      builder.create<sv::VerbatimOp>(
          uloc, "assign " + path + " = " + srcRef +
                    module.portNames()[blockArg.getArgNumber()]
                        .cast<StringAttr>()
                        .getValue() +
                    ";");
    } else
      builder.create<sv::VerbatimOp>(uloc, "assign " + path + " = " + srcRef +
                                               leafValue.getDefiningOp()
                                                   ->getAttr("name")
                                                   .cast<StringAttr>()
                                                   .getValue() +
                                               ";");

    auto width = leafValue.getType().cast<FIRRTLType>().getBitWidthOrSentinel();
    // The name will exist if this is a ground type or a ground type of a bundle
    // type.  There is expected to be no name if this is a ground type of a
    // vector type.
    if (buildIFace)
      tpe.push_back(GroundKind(name ? name.getValue() : "", width));
    return true;
  }
  case Element::String: {
    if (!name) {
      llvm::errs() << "string element missing 'name'\n";
      return false;
    }
    tpe.push_back(StringKind(name.getValue()));
    return true;
  }
  case Element::Boolean: {
    if (!name) {
      llvm::errs() << "boolean element missing 'name'\n";
      return false;
    }
    tpe.push_back(BooleanKind(name.getValue()));
    return true;
  }
  case Element::Integer: {
    if (!name) {
      llvm::errs() << "integer element missing 'name'\n";
      return false;
    }
    tpe.push_back(IntegerKind(name.getValue()));
    return true;
  }
  case Element::Double: {
    if (!name) {
      llvm::errs() << "double element missing 'name'\n";
      return false;
    }
    tpe.push_back(DoubleKind(name.getValue()));
    return true;
  }
  case Element::Error: {
    llvm::errs() << "unknown field class name '" << clazz.getValue() << "'\n";
    return false;
  }
  }
}

/// Traverse an Annotation that is an AugmentedBundleType.  During traversal,
/// construct any discovered SystemVerilog interfaces.  If this is the root
/// interface, instantiate that interface in the parent.  Recurse into fields of
/// the AugmentedBundleType to construct nested interfaces and generate
/// stringy-typed SystemVerilog hierarchical references to drive the interface.
/// Returns false on any failure and true on success.
///
/// This is a normal tree traversal with dual effects.  This traversal will
/// always generate the hierarchical refernces (as these are located at the
/// leaves of the Annotation).  However, this only generates the interface if
/// the `buildIFace` parameter is true.  This is done to prevent interface
/// construction for every element of a vector (and instead just create one
/// interface for the whole vector).
bool GrandCentralPass::traverseBundle(Annotation anno, IntegerAttr id,
                                      Twine path, bool buildIFace) {

  BundleType bundle;
  if (auto maybeBundle = BundleType::fromAnno(anno))
    bundle = maybeBundle.getValue();
  else
    return false;

  // Set the ID if it is not already set and verify that everything is setup for
  // further processing:
  //   1. If the ID isn't set, then this must be the top-level BundleType.
  //      Ensure that it is by checking that the BundleType has an ID.
  //   2. A parent must have been found in the circuit.
  //   3. A companion must have been found in the circuit.
  if (!id) {
    id = bundle.id;
    if (!id) {
      llvm::errs() << "missing 'id' in root-level BundleType: "
                   << anno.getDict() << "\n";
      return false;
    }
    if (parentIDMap.count(id) == 0) {
      llvm::errs() << "no parent found with 'id' value '"
                   << id.getValue().getZExtValue() << "'\n";
      return false;
    }
    if (companionIDMap.count(id) == 0) {
      llvm::errs() << "no companion found with 'id' value '"
                   << id.getValue().getZExtValue() << "'\n";
      return false;
    }
  }

  auto builder = OpBuilder::atBlockEnd(getOperation().getBody());
  if (buildIFace) {
    builder.setInsertionPointToEnd(getOperation().getBody());
    auto loc = getOperation().getLoc();
    sv::InterfaceOp iface =
        builder.create<sv::InterfaceOp>(loc, bundle.defName.getValue());
    iface->setAttr("output_file",
                   hw::OutputFileAttr::get(
                       builder.getStringAttr(getOutputDirectory()),
                       builder.getStringAttr(bundle.defName.getValue() + ".sv"),
                       builder.getBoolAttr(true), builder.getBoolAttr(true),
                       builder.getContext()));

    // If this is the root interface, then it needs to be instantiated in the
    // parent.
    if (bundle.isRoot()) {
      builder.setInsertionPointToEnd(
          parentIDMap.lookup(id).second.getBodyBlock());
      auto instance = builder.create<sv::InterfaceInstanceOp>(
          loc, iface.getInterfaceType(), companionIDMap.lookup(id).name,
          builder.getStringAttr("__" + companionIDMap.lookup(id).name + "_" +
                                bundle.defName.getValue() + "__"));

      // If there was no bind file passed in, then we're not supposed to
      // extract this.  Delete the annotation and continue.
      if (maybeExtractInfo) {

        instance->setAttr("doNotPrint", builder.getBoolAttr(true));
        builder.setInsertionPointToStart(
            instance->getParentOfType<ModuleOp>().getBody());
        auto bind = builder.create<sv::BindInterfaceOp>(
            loc, builder.getSymbolRefAttr(instance.sym_name().getValue()));
        bind->setAttr(
            "output_file",
            hw::OutputFileAttr::get(
                builder.getStringAttr(""),
                builder.getStringAttr(maybeExtractInfo.getValue().bindFilename),
                builder.getBoolAttr(true), builder.getBoolAttr(true),
                bind.getContext()));
      }
    }

    builder.setInsertionPointToEnd(cast<sv::InterfaceOp>(iface).getBodyBlock());
  }

  for (auto element : bundle.elements) {
    FieldType field;
    if (auto maybeField = FieldType::fromAttr(element))
      field = maybeField.getValue();
    else
      return false;

    SmallVector<Element> tpe;
    if (!traverseField(element, tpe, id,
                       path.isTriviallyEmpty()
                           ? bundle.defName.getValue() + "." +
                                 field.name.getValue()
                           : path + "." + field.name.getValue(),
                       buildIFace))
      return false;

    if (buildIFace)
      buildInterfaceSignal(tpe, builder, field.name.getValue(),
                           field.description);
  }

  return true;
}

void GrandCentralPass::buildInterfaceSignal(ArrayRef<Element> elements,
                                            OpBuilder &builder, StringRef name,
                                            StringAttr description) {

  // This is walking a sequency of element types to build up a representation of
  // an interface signal.  This is either an actual interface signal type or a
  // string representing something that is another interface.  This is using a
  // string for a nested interface due to existing limitations around doing this
  // in the SV dialect (see: https://github.com/llvm/circt/issues/1171).
  std::string str;
  llvm::raw_string_ostream s(str);
  Type tpe;
  bool isIface = false;
  for (auto element : llvm::reverse(elements)) {
    TypeSwitch<Element *>(&element)
        .Case<GroundKind>(
            [&](auto a) { tpe = builder.getIntegerType(a->getWidth()); })
        .Case<VectorKind>([&](auto a) {
          if (!str.empty()) {
            s << "[" << a->getDepth() << "]";
            return;
          }
          tpe = hw::UnpackedArrayType::get(tpe, a->getDepth());
        })
        .Case<BundleKind>([&](auto a) {
          isIface = true;
          s << a->name << " " << name;
        })
        .Case<StringKind>([&](auto a) {
          assert(elements.size() == 1);
          s << "// " << a->name << " = "
            << "<unsupported string type>";
        })
        .Case<BooleanKind>([&](auto a) {
          assert(elements.size() == 1);
          s << "// " << a->name << " = "
            << "<unsupported boolean type>";
        })
        .Case<IntegerKind>([&](auto a) {
          assert(elements.size() == 1);
          s << "// " << a->name << " = "
            << "<unsupported integer type>";
        })
        .Case<DoubleKind>([&](auto a) {
          assert(elements.size() == 1);
          s << "// " << a->name << " = "
            << "<unsupported double type>";
        });
  }

  // If this is an interface, then add a "()" at the end.  This is enabling
  // vectors of interfaces which have to be constructed like:
  //
  //     Foo Foo[2][4][8]()
  if (isIface)
    s << "()";

  // Construct the type, included an optional description.
  auto uloc = builder.getUnknownLoc();
  if (description)
    builder.create<sv::VerbatimOp>(uloc,
                                   ("// " + description.getValue()).str());
  if (str.empty()) {
    builder.create<sv::InterfaceSignalOp>(getOperation().getLoc(), name, tpe);
    return;
  }

  s << ";";
  builder.create<sv::VerbatimOp>(uloc, str);
}

FModuleOp GrandCentralPass::getEnclosingModule(Value value) {
  if (auto blockArg = value.dyn_cast<BlockArgument>())
    return cast<FModuleOp>(blockArg.getOwner()->getParentOp());

  auto *op = value.getDefiningOp();
  if (InstanceOp instance = dyn_cast<InstanceOp>(op))
    return cast<FModuleOp>(instance.getReferencedModule());

  return op->getParentOfType<FModuleOp>();
}

void GrandCentralPass::runOnOperation() {
  CircuitOp circuitOp = getOperation();

  AnnotationSet annotations(circuitOp);
  if (annotations.empty())
    return;

  // Utility that acts like emitOpError, but does _not_ include a note.  The
  // note in emitOpError includes the entire op which means the **ENTIRE**
  // FIRRTL circuit.  This doesn't communicate anything useful to the user
  // other than flooding their terminal.
  auto emitCircuitError = [&circuitOp](StringRef message = {}) {
    return emitError(circuitOp.getLoc(), message);
  };

  // Pull out the extraction info if it exists.
  bool removalError = false;
  annotations.removeAnnotations([&](Annotation anno) {
    if (anno.isClass(
            "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation")) {
      if (maybeExtractInfo.hasValue()) {
        emitCircuitError("more than one 'ExtractGrandCentralAnnotation' was "
                         "found, but exactly one must be provided");
        removalError = true;
        return false;
      }

      maybeExtractInfo = {anno.getMember<StringAttr>("directory").getValue(),
                          anno.getMember<StringAttr>("filename").getValue()};
      return false;
    }
    return false;
  });

  if (removalError)
    return signalPassFailure();

  llvm::errs() << "Extraction Info:\n";
  if (maybeExtractInfo)
    llvm::errs() << "  directory: " << maybeExtractInfo.getValue().directory
                 << "\n"
                 << "  filename: " << maybeExtractInfo.getValue().bindFilename
                 << "\n";
  else
    llvm::errs() << "  <none>\n";

  // Setup the builder to create ops _inside the FIRRTL circuit_.  This is
  // necessary because interfaces and interface instances are created.
  // Instances link to their definitions via symbols and we don't want to
  // break this.
  auto builder = OpBuilder::atBlockEnd(circuitOp.getBody());

  SymbolTable symbolTable(circuitOp);

  removalError = false;
  circuitOp.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<RegOp, RegResetOp, WireOp, NodeOp>([&](auto op) {
          AnnotationSet::removeAnnotations(op, [&](Annotation annotation) {
            if (!annotation.isClass(
                    "sifive.enterprise.grandcentral.AugmentedGroundType"))
              return false;

            auto id = annotation.getMember<IntegerAttr>("id");
            if (!id) {
              op.emitOpError()
                  << "contained a malformed "
                     "'sifive.enterprise.grandcentral.AugmentedGroundType' "
                     "annotation that did not contain an 'id' field";
              removalError = true;
              return false;
            }

            leafMap.insert({id, op.getResult()});

            return true;
          });
        })
        .Case<InstanceOp>([&](auto op) {
          /* Populate the leafMap for ports. */
        })
        .Case<FModuleOp>([&](FModuleOp op) {
          // Handle annotations on the ports.
          AnnotationSet::removePortAnnotations(
              op, [&](unsigned i, Annotation annotation) {
                if (!annotation.isClass(
                        "sifive.enterprise.grandcentral.AugmentedGroundType"))
                  return false;

                auto id = annotation.getMember<IntegerAttr>("id");
                if (!id) {
                  op.emitOpError()
                      << "contained a malformed "
                         "'sifive.enterprise.grandcentral.AugmentedGroundType' "
                         "annotation that did not contain an 'id' field";
                  removalError = true;
                  return false;
                }

                leafMap.insert({id, op.getArgument(i)});

                return true;
              });

          // Handle annotations on the module.
          AnnotationSet::removeAnnotations(op, [&](Annotation annotation) {
            if (!annotation.isClass(
                    "sifive.enterprise.grandcentral.ViewAnnotation"))
              return false;
            auto tpe = annotation.getMember<StringAttr>("type");
            auto id = annotation.getMember<IntegerAttr>("id");
            auto name = annotation.getMember<StringAttr>("name");
            if (!tpe) {
              op.emitOpError()
                  << "has a malformed "
                     "'sifive.enterprise.grandcentral.ViewAnnotation' that "
                     "did "
                     "not contain a 'type' field with a 'StringAttr' value";
              goto FModuleOp_error;
            }
            if (!id) {
              op.emitOpError()
                  << "has a malformed "
                     "'sifive.enterprise.grandcentral.ViewAnnotation' that "
                     "did "
                     "not contain an 'id' field with an 'IntegerAttr' value";
              goto FModuleOp_error;
            }
            if (!name) {
              op.emitOpError()
                  << "has a malformed "
                     "'sifive.enterprise.grandcentral.ViewAnnotation' that "
                     "did "
                     "not contain a 'name' field with a 'StringAttr' value";
              goto FModuleOp_error;
            }
            if (tpe.getValue() == "companion") {
              auto mapping = builder.create<FModuleOp>(
                  circuitOp.getLoc(),
                  builder.getStringAttr(name.getValue() + "_mapping"),
                  SmallVector<ModulePortInfo>({}));
              auto *ctx = builder.getContext();
              mapping->setAttr(
                  "output_file",
                  hw::OutputFileAttr::get(
                      StringAttr::get(ctx, getOutputDirectory()),
                      StringAttr::get(ctx, mapping.getName() + ".sv"),
                      BoolAttr::get(ctx, true), BoolAttr::get(ctx, true), ctx));
              companionIDMap.insert({id, {name.getValue(), op, mapping}});

              auto instances = symbolTable.getSymbolUses(op, circuitOp);
              if (!instances ||
                  std::distance(instances.getValue().begin(),
                                instances.getValue().end()) == 0) {
                op.emitOpError() << "is marked as a GrandCentral 'companion', "
                                    "but is never instantiated";
                return false;
              }

              if (std::distance(instances.getValue().begin(),
                                instances.getValue().end()) != 1) {
                auto diag = op.emitOpError()
                            << "is marked as a GrandCentral 'companion', but "
                               "it is instantiated more than once";
                for (auto instance : instances.getValue())
                  diag.attachNote(instance.getUser()->getLoc())
                      << "parent is instantiated here";
                return false;
              }

              auto instance =
                  cast<InstanceOp>((*(instances.getValue().begin())).getUser());

              // Instantiate the mapping module inside the companion.
              builder.setInsertionPointToEnd(op.getBodyBlock());
              builder.create<InstanceOp>(circuitOp.getLoc(),
                                         SmallVector<Type>({}),
                                         mapping.getName(), mapping.getName());

              // If an extraction annotation was found then extract the
              // interface.
              if (maybeExtractInfo) {
                instance->setAttr("lowerToBind", BoolAttr::get(ctx, true));
                instance->setAttr(
                    "output_file",
                    hw::OutputFileAttr::get(
                        StringAttr::get(ctx, ""),
                        StringAttr::get(
                            ctx, maybeExtractInfo.getValue().bindFilename),
                        /*exclude_from_filelist=*/
                        BoolAttr::get(ctx, true),
                        /*exclude_replicated_ops=*/
                        BoolAttr::get(ctx, true), ctx));
                op->setAttr("output_file",
                            hw::OutputFileAttr::get(
                                StringAttr::get(
                                    ctx, maybeExtractInfo.getValue().directory),
                                StringAttr::get(ctx, op.getName() + ".sv"),
                                /*exclude_from_filelist=*/
                                BoolAttr::get(ctx, true),
                                /*exclude_replicated_ops=*/
                                BoolAttr::get(ctx, true), ctx));
              }
              return true;
            }
            if (tpe.getValue() == "parent") {
              auto instances = symbolTable.getSymbolUses(op, circuitOp);
              if (!instances ||
                  std::distance(instances.getValue().begin(),
                                instances.getValue().end()) == 0) {
                op.emitOpError() << "is marked as a GrandCentral 'parent', but "
                                    "is never instantiated";
                return false;
              }

              if (std::distance(instances.getValue().begin(),
                                instances.getValue().end()) != 1) {
                auto diag = op.emitOpError()
                            << "is marked as a GrandCentral 'parent', but it "
                               "is instantiated more than once";
                for (auto instance : instances.getValue())
                  diag.attachNote(instance.getUser()->getLoc())
                      << "parent is instantiated here";
                return false;
              }

              auto instance =
                  cast<InstanceOp>((*(instances.getValue().begin())).getUser());

              parentIDMap.insert({id, {instance, cast<FModuleOp>(op)}});
              return true;
            }
            op.emitOpError()
                << "has a 'sifive.enterprise.grandcentral.ViewAnnotation' "
                   "with "
                   "an unknown 'type' field";
          FModuleOp_error:
            removalError = true;
            return false;
          });
        });
  });

  if (removalError)
    return signalPassFailure();

  // Check that the parent and the companion both exist.
  for (auto a : companionIDMap) {
    if (parentIDMap.count(a.first) == 0) {
      emitCircuitError()
          << "contains a 'companion' with id '"
          << a.first.cast<IntegerAttr>().getValue().getZExtValue()
          << "', but does not contain a GrandCentral 'parent' with the same "
             "id";
      return signalPassFailure();
    }
  }
  for (auto a : parentIDMap) {
    if (companionIDMap.count(a.first) == 0) {
      emitCircuitError()
          << "contains a 'parent' with id '"
          << a.first.cast<IntegerAttr>().getValue().getZExtValue()
          << "', but does not contain a GrandCentral 'companion' "
             "with the same id";
      return signalPassFailure();
    }
  }

  llvm::errs() << "companionIDMap:\n";
  for (auto a : companionIDMap)
    llvm::errs() << "  - " << a.first.cast<IntegerAttr>().getValue() << ": "
                 << a.second.companion.getName() << " -> " << a.second.name
                 << "\n";

  llvm::errs() << "parentIDMap:\n";
  for (auto a : parentIDMap)
    llvm::errs() << "  - " << a.first.cast<IntegerAttr>().getValue() << ": "
                 << a.second.first.name() << ":" << a.second.second.getName()
                 << "\n";

  llvm::errs() << "leafMap:\n";
  for (auto a : leafMap) {
    if (auto blockArg = a.second.dyn_cast<BlockArgument>()) {
      FModuleOp module = cast<FModuleOp>(blockArg.getOwner()->getParentOp());
      llvm::errs() << "  - " << a.first.cast<IntegerAttr>().getValue() << ": "
                   << module.getName() + ">" +
                          module.portNames()[blockArg.getArgNumber()]
                              .cast<StringAttr>()
                              .getValue()
                   << "\n";
    } else {
      llvm::errs() << "  - " << a.first.cast<IntegerAttr>().getValue() << ": "
                   << a.second.getDefiningOp()
                          ->getAttr("name")
                          .cast<StringAttr>()
                          .getValue()
                   << "\n";
    }
  }

  InstancePaths instancePaths(getAnalysis<InstanceGraph>());

  instancePathsRef = &instancePaths;

  // Generate everything.
  annotations.removeAnnotations([&](Annotation anno) {
    if (!anno.isClass("sifive.enterprise.grandcentral.AugmentedBundleType"))
      return false;

    auto id = anno.getMember<IntegerAttr>("id");
    if (!traverseBundle(anno, {}, companionIDMap.lookup(id).name, true)) {
      emitCircuitError(
          "'firrtl.circuit' op contained an 'AugmentedBundleType' "
          "Annotation which did not conform to the expected format")
              .attachNote()
          << "the problematic 'AugmentedBundleType' is: '" << anno.getDict()
          << "'";
      removalError = true;
      return false;
    }

    return true;
  });

  // Signal pass failure if any errors were found while examining circuit
  // annotations.
  if (removalError)
    return signalPassFailure();

  annotations.applyToOperation(circuitOp);
}

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> circt::firrtl::createGrandCentralPass() {
  return std::make_unique<GrandCentralPass>();
}
