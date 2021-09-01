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
using llvm::Optional;
using llvm::PointerSumType;
using llvm::PointerSumTypeMember;
using llvm::SmallString;
using llvm::StringSwitch;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {

namespace Kind {
enum Kind { Error, Ground, Vector, Bundle, Unsupported };
} // namespace Kind

/// Adds type information to a DictionaryAttr by using PointerSumType.  These
/// are used to add type information to an unstructured annotation to certify
/// that it is of a specific type.  This allows verification to be separated
/// from use.
typedef PointerSumType<
    Kind::Kind, PointerSumTypeMember<Kind::Error, DictionaryAttr *>,
    PointerSumTypeMember<Kind::Ground, DictionaryAttr *>,
    PointerSumTypeMember<Kind::Vector, DictionaryAttr *>,
    PointerSumTypeMember<Kind::Bundle, DictionaryAttr *>,
    PointerSumTypeMember<Kind::Unsupported, DictionaryAttr *>>
    SumType;

/// Convert a DictionaryAttr to an AugmentedType.  This handles all
/// validation logic related to checking the DictionaryAttr.  The returned
/// type, so long as it is not Error, can be trivially converted to an
/// AugmentedType struct.
static SumType fromDict(DictionaryAttr *dict) {
  auto clazz = dict->getAs<StringAttr>("class");
  if (!clazz) {
    llvm::errs() << "missing 'class' key in " << dict << "\n";
    return SumType::create<Kind::Error>(dict);
  }
  auto classBase = clazz.getValue();
  classBase.consume_front("sifive.enterprise.grandcentral.Augmented");

  auto kind = StringSwitch<Kind::Kind>(classBase)
                  .Case("BundleType", Kind::Bundle)
                  .Case("VectorType", Kind::Vector)
                  .Case("GroundType", Kind::Ground)
                  .Case("StringType", Kind::Unsupported)
                  .Case("BooleanType", Kind::Unsupported)
                  .Case("IntegerType", Kind::Unsupported)
                  .Case("DoubleType", Kind::Unsupported)
                  .Case("LiteralType", Kind::Unsupported)
                  .Case("DeletedType", Kind::Unsupported)
                  .Default(Kind::Error);

  switch (kind) {
  case Kind::Bundle:
    if (dict->getAs<StringAttr>("defName") &&
        dict->getAs<ArrayAttr>("elements"))
      return SumType::create<Kind::Bundle>(dict);
    break;
  case Kind::Vector:
    if (dict->getAs<StringAttr>("name") && dict->getAs<ArrayAttr>("elements"))
      return SumType::create<Kind::Vector>(dict);
    break;
  case Kind::Ground:
    if (dict->getAs<IntegerAttr>("id") && dict->getAs<StringAttr>("name"))
      return SumType::create<Kind::Ground>(dict);
    break;
  case Kind::Unsupported:
    if (dict->getAs<StringAttr>("name"))
      return SumType::create<Kind::Unsupported>(dict);
    break;
  case Kind::Error:
    break;
  }

  assert(false && "wasn't able to build SumType from dict");
  return SumType::create<Kind::Error>(dict);
}

struct Bundle {
  IntegerAttr id;
  StringAttr defName;
  ArrayAttr elements;

  bool isRoot() { return id != nullptr; }

  static Bundle fromSumType(SumType a) {
    auto *dict = a.get<Kind::Bundle>();
    assert(dict && "input must be a Bundle type");
    return Bundle({dict->getAs<IntegerAttr>("id"),
                   dict->getAs<StringAttr>("defName"),
                   dict->getAs<ArrayAttr>("elements")});
  }
};

struct Vector {
  StringAttr name;
  ArrayAttr elements;

  static Vector fromSumType(SumType a) {
    auto *dict = a.get<Kind::Vector>();
    assert(dict && "input must be a Vector type");
    return Vector(
        {dict->getAs<StringAttr>("name"), dict->getAs<ArrayAttr>("elements")});
  }
};

struct Ground {
  IntegerAttr id;
  StringAttr name;

  static Ground fromSumType(SumType a) {
    auto *dict = a.get<Kind::Ground>();
    assert(dict && "input must be a Ground type");
    return Ground(
        {dict->getAs<IntegerAttr>("id"), dict->getAs<StringAttr>("name")});
  }
};

struct Unsupported {
  StringAttr clazz;
  StringAttr name;
  DictionaryAttr *underlying;

  static Unsupported fromSumType(SumType a) {
    auto *dict = a.get<Kind::Unsupported>();
    assert(dict && "input must be an Unsupported type");
    return Unsupported({dict->getAs<StringAttr>("class"),
                        dict->getAs<StringAttr>("name"), dict});
  }
};

struct CircuitNamespace {

private:
  StringSet<> internal;
  CircuitNamespace(StringSet<> internal) : internal(internal) {}

public:
  CircuitNamespace(CircuitOp op) {
    op.walk([&](Operation *op) {
      if (auto symbol = op->getAttrOfType<StringAttr>("sym_name")) {
        internal.insert(symbol.getValue());
      }
      return WalkResult::skip();
    });
  }

private:
  SmallString<0> newNameImpl(StringRef name) {
    if (!internal.count(name)) {
      internal.insert(name);
      return name;
    }
    size_t i = 0;
    SmallString<0> tryName;
    do {
      tryName = (name + "_" + Twine(i++)).str();
    } while (internal.count(tryName));
    return tryName;
  }

public:
  SmallString<0> newName(StringRef name) { return newNameImpl(name); }
  SmallString<0> newName(Twine name) { return newNameImpl(name.str()); }
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
  DenseMap<Attribute, Value> leafMap;

  // Mapping of ID to parent instance and module.
  DenseMap<Attribute, std::pair<InstanceOp, FModuleOp>> parentIDMap;

  // Mapping of ID to companion module.
  DenseMap<Attribute, CompanionInfo> companionIDMap;

  bool traverseField(SumType field, IntegerAttr id, Twine path);

  std::variant<SmallString<0>, SumType, Type>
  computeField(SumType field, IntegerAttr id, Twine path);

  bool walkBundle(Bundle bundle, IntegerAttr id = {}, Twine path = {});

  Optional<sv::InterfaceOp> traverseBundle(Bundle bundle, IntegerAttr id = {},
                                           Twine path = {});

  FModuleLike getEnclosingModule(Value value);

  // Inforamtion about how the circuit should be extracted.  This will be
  // non-empty if an extraction annotation is found.
  Optional<ExtractionInfo> maybeExtractInfo = None;

  StringRef getOutputDirectory() {
    return maybeExtractInfo ? maybeExtractInfo.getValue().directory : ".";
  }

  InstancePaths *instancePathsRef;

  SymbolTable *symbolTableRef;

  CircuitNamespace *circuitNamespaceRef;
};

} // namespace

bool GrandCentralPass::traverseField(SumType field, IntegerAttr id,
                                     Twine path) {
  switch (field.getTag()) {
  case Kind::Ground: {
    auto ground = Ground::fromSumType(field);

    // TODO: move this into the sumtype verification.
    auto leafValue = leafMap.lookup(ground.id);
    if (!leafValue) {
      llvm::errs() << "missing 'leafValue' for GroundType with id " << ground.id
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
    return true;
  }
  case Kind::Vector: {
    auto vector = Vector::fromSumType(field);
    bool notFailed = true;
    for (size_t i = 0, e = vector.elements.size(); i != e; ++i) {
      auto dict = vector.elements[i].dyn_cast<DictionaryAttr>();
      if (!dict)
        return false;
      auto sumType = fromDict(&dict);
      notFailed &= traverseField(sumType, id, path + "[" + Twine(i) + "]");
    }
    return notFailed;
  }
  case Kind::Bundle: {
    auto bundle = Bundle::fromSumType(field);
    return walkBundle(bundle, id, path);
  }
  case Kind::Unsupported: {
    return true;
  }
  default:
    assert(false && "unhandled");
  }
}

std::variant<SmallString<0>, SumType, Type>
GrandCentralPass::computeField(SumType field, IntegerAttr id, Twine path) {
  switch (field.getTag()) {
  case Kind::Ground: {
    auto ground = Ground::fromSumType(field);

    // TODO: move this into the sumtype verification.
    auto leafValue = leafMap.lookup(ground.id);
    if (!leafValue) {
      llvm::errs() << "missing 'leafValue' for GroundType with id " << ground.id
                   << "\n";
      assert(false);
    }

    // Traverse to generate mappings.
    traverseField(field, id, path);

    return IntegerType::get(
        getOperation().getContext(),
        leafValue.getType().cast<FIRRTLType>().getBitWidthOrSentinel());
  }
  case Kind::Vector: {
    bool notFailed = true;
    std::variant<SmallString<0>, SumType, Type> elementType;
    auto vector = Vector::fromSumType(field);

    for (size_t i = 0, e = vector.elements.size(); i != e; ++i) {
      auto dict = vector.elements[i].dyn_cast<DictionaryAttr>();
      assert(dict && "not a dict");
      auto sumType = fromDict(&dict);
      if (i == 0)
        elementType = computeField(sumType, id, path + "[" + Twine(i) + "]");
      else
        notFailed &= traverseField(sumType, id, path + "[" + Twine(i) + "]");
    }

    if (auto *tpe = std::get_if<Type>(&elementType))
      return hw::UnpackedArrayType::get(*tpe,
                                        vector.elements.getValue().size());
    auto str = std::get<SmallString<0>>(elementType);
    str.append(("[" + Twine(vector.elements.getValue().size()) + "]").str());
    return str;
  }
  case Kind::Bundle: {
    auto bundle = Bundle::fromSumType(field);
    auto iface = traverseBundle(bundle, id, path);
    assert(iface && iface.getValue());
    return SmallString<0>(
        (iface.getValue().getName() + " " + iface.getValue().getName()).str());
  }
  case Kind::Unsupported: {
    return field;
  }
  default:
    assert(false && "unhandled");
  }
}

bool GrandCentralPass::walkBundle(Bundle bundle, IntegerAttr id, Twine path) {
  bool anyFailed = true;
  for (auto element : bundle.elements) {
    DictionaryAttr field = element.dyn_cast<DictionaryAttr>();
    if (!field)
      return false;
    auto name = field.getAs<StringAttr>("name");
    if (!name)
      name = field.getAs<StringAttr>("defName");
    anyFailed &=
        traverseField(fromDict(&field), id,
                      path.isTriviallyEmpty()
                          ? bundle.defName.getValue() + "." + name.getValue()
                          : path + "." + name.getValue());
  }

  return anyFailed;
}

/// Traverse an Annotation that is an AugmentedBundleType.  During
/// traversal, construct any discovered SystemVerilog interfaces.  If this
/// is the root interface, instantiate that interface in the parent. Recurse
/// into fields of the AugmentedBundleType to construct nested interfaces
/// and generate stringy-typed SystemVerilog hierarchical references to
/// drive the interface. Returns false on any failure and true on success.
Optional<sv::InterfaceOp>
GrandCentralPass::traverseBundle(Bundle bundle, IntegerAttr id, Twine path) {
  auto builder = OpBuilder::atBlockEnd(getOperation().getBody());
  sv::InterfaceOp iface;
  builder.setInsertionPointToEnd(getOperation().getBody());
  auto loc = getOperation().getLoc();
  auto iFaceName = circuitNamespaceRef->newName(bundle.defName.getValue());
  iface = builder.create<sv::InterfaceOp>(loc, iFaceName);
  iface->setAttr(
      "output_file",
      hw::OutputFileAttr::get(builder.getStringAttr(getOutputDirectory()),
                              builder.getStringAttr(iFaceName + ".sv"),
                              builder.getBoolAttr(true),
                              builder.getBoolAttr(true), builder.getContext()));

  // If this is the root interface, then it needs to be instantiated in the
  // parent.
  if (bundle.isRoot()) {
    builder.setInsertionPointToEnd(
        parentIDMap.lookup(id).second.getBodyBlock());
    auto symbolName =
        circuitNamespaceRef->newName(("__" + companionIDMap.lookup(id).name +
                                      "_" + bundle.defName.getValue() + "__"));
    llvm::errs() << "Try to add symbol: " << symbolName << "\n";
    auto instance = builder.create<sv::InterfaceInstanceOp>(
        loc, iface.getInterfaceType(), companionIDMap.lookup(id).name,
        builder.getStringAttr(symbolName));

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

  for (auto element : bundle.elements) {
    DictionaryAttr field = element.dyn_cast<DictionaryAttr>();
    if (!field)
      return None;
    auto sumType = fromDict(&field);

    auto name = field.getAs<StringAttr>("name");
    if (!name)
      name = field.getAs<StringAttr>("defName");
    auto elementType =
        computeField(sumType, id,
                     path.isTriviallyEmpty()
                         ? bundle.defName.getValue() + "." + name.getValue()
                         : path + "." + name.getValue());

    auto uloc = builder.getUnknownLoc();
    auto description = field.getAs<StringAttr>("description");
    if (description)
      builder.create<sv::VerbatimOp>(uloc,
                                     ("// " + description.getValue()).str());
    if (auto *str = std::get_if<SmallString<0>>(&elementType)) {
      builder.create<sv::VerbatimOp>(uloc, str->str() + "();");
      continue;
    }

    if (auto *ptr = std::get_if<SumType>(&elementType)) {
      auto unsupported = Unsupported::fromSumType(*ptr);
      auto classBase = unsupported.clazz.getValue();
      classBase.consume_front("sifive.enterprise.grandcentral.Augmented");
      if (classBase == "StringType") {
        builder.create<sv::VerbatimOp>(uloc,
                                       "// " + unsupported.name.getValue() +
                                           " = <unsupported string type>;");
      } else if (classBase == "BooleanType") {
        builder.create<sv::VerbatimOp>(uloc,
                                       "// " + unsupported.name.getValue() +
                                           " = <unsupported boolean type>;");
      } else if (classBase == "IntegerType") {
        builder.create<sv::VerbatimOp>(uloc,
                                       "// " + unsupported.name.getValue() +
                                           " = <unsupported integer type>;");
      } else if (classBase == "DoubleType") {
        builder.create<sv::VerbatimOp>(uloc,
                                       "// " + unsupported.name.getValue() +
                                           " = <unsupported double type>;");
      }
      continue;
    }

    builder.create<sv::InterfaceSignalOp>(uloc, name.getValue(),
                                          std::get<Type>(elementType));
  }

  return iface;
}

FModuleLike GrandCentralPass::getEnclosingModule(Value value) {
  if (auto blockArg = value.dyn_cast<BlockArgument>())
    return cast<FModuleOp>(blockArg.getOwner()->getParentOp());

  auto *op = value.getDefiningOp();
  if (InstanceOp instance = dyn_cast<InstanceOp>(op))
    return instance.getReferencedModule();

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
  symbolTableRef = &symbolTable;

  CircuitNamespace circuitNamespace(circuitOp);
  circuitNamespaceRef = &circuitNamespace;

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
              builder.setInsertionPointToEnd(circuitOp.getBody());
              auto mappingName =
                  circuitNamespaceRef->newName(name.getValue() + "_mapping");
              auto mapping = builder.create<FModuleOp>(
                  circuitOp.getLoc(), builder.getStringAttr(mappingName),
                  SmallVector<ModulePortInfo>({}));
              symbolTable.insert(mapping);
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

    auto dict = anno.getDict();
    SumType sumType = fromDict(&dict);
    if (!sumType.is<Kind::Bundle>())
      return false;

    auto bundle = Bundle::fromSumType(sumType);
    if (!bundle.isRoot()) {
      llvm::errs() << "missing 'id' in root-level BundleType: " << dict << "\n";
      removalError = true;
      return false;
    }
    if (parentIDMap.count(bundle.id) == 0) {
      llvm::errs() << "no parent found with 'id' value '"
                   << bundle.id.getValue().getZExtValue() << "'\n";
      removalError = true;
      return false;
    }
    if (companionIDMap.count(bundle.id) == 0) {
      llvm::errs() << "no companion found with 'id' value '"
                   << bundle.id.getValue().getZExtValue() << "'\n";
      removalError = true;
      return false;
    }
    if (!traverseBundle(bundle, bundle.id,
                        companionIDMap.lookup(bundle.id).name)) {
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
