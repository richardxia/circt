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
  enum ElementKind { Bundle, Vector, Ground };

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

class GroundKind : public Element {
public:
  GroundKind(StringRef name, uint64_t width)
      : Element({ElementKind::Ground, name, width}) {}
  static bool classof(const Element *e) {
    return e->getKind() == ElementKind::Ground;
  }
  uint64_t getWidth() const { return storage; }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const Element element) {
  TypeSwitch<const Element *>(&element)
      .Case<BundleKind>([&](auto a) { os << "BUNDLE -> " << a->name; })
      .Case<VectorKind>([&](auto a) {
        os << "VECTOR -> name: " << a->name << ", depth: " << a->getDepth();
      })
      .Case<GroundKind>([&](auto a) {
        os << "GROUND -> name: " << a->name << ", width: " << a->getWidth();
      });
  return os;
};

namespace {
/// Mutable store of information about an Element in an interface.  This is
/// derived from information stored in the "elements" field of an
/// "AugmentedBundleType".  This is updated as more information is known about
/// an Element.
struct ElementInfo {
  /// Encodes the "tpe" of an element.  This is called "Kind" to avoid
  /// overloading the meaning of "Type" (which also conflicts with mlir::Type).
  enum Kind {
    Error = -1,
    Ground,
    Vector,
    Bundle,
    String,
    Boolean,
    Integer,
    Double
  };
  /// The "tpe" field indicating if this element of the interface is a ground
  /// type, a vector type, or a bundle type.  Bundle types are nested
  /// interfaces.
  Kind tpe;
  /// A string description that will show up as a comment in the output Verilog.
  StringRef description;
  /// The width of this interface.  This is only non-negative for ground or
  /// vector types.
  int32_t width = -1;
  /// The depth of the interface.  This is one for ground types and greater
  /// than one for vector types.
  uint32_t depth = 0;
  /// Indicate if this element was found in the circuit.
  bool found = false;
  /// Trakcs location information about what was used to build this element.
  SmallVector<Location> locations = {};
  /// The FIRRTL operation that this is supposed to be connected to.  This is
  /// null if no operation was found or has yet been found.
  Operation *op = {};
  /// True if this is a ground or vector type and it was not (statefully) found.
  /// This indicates that an interface element, which is composed of ground and
  /// vector types, found no matching, annotated components in the circuit.
  bool isMissing() { return !found && (tpe == Ground || tpe == Vector); }
};

/// Stores a decoded Grand Central AugmentedField
struct AugmentedField {
  /// The name of the field.
  StringRef name;
  /// An optional descripton that the user provided for the field.  This should
  /// become a comment in the Verilog.
  StringRef description;
  /// The "type" of the field.
  ElementInfo::Kind tpe;
  /// An optional global identifier.
  IntegerAttr id;
};

/// Stores a decoded Grand Central AugmentedBundleType.
struct AugmentedBundleType {
  /// The name of the interface.
  StringRef defName;
  /// The elements that make up the body of the interface.
  SmallVector<AugmentedField> elements;
  /// An optional global identifier.
  IntegerAttr id;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ElementInfo::Kind tpe) {
  switch (tpe) {
  case ElementInfo::Error:
    os << "error";
    break;
  case ElementInfo::Ground:
    os << "ground";
    break;
  case ElementInfo::Vector:
    os << "vector";
    break;
  case ElementInfo::Bundle:
    os << "bundle";
    break;
  case ElementInfo::String:
    os << "string";
    break;
  case ElementInfo::Boolean:
    os << "boolean";
    break;
  case ElementInfo::Integer:
    os << "integer";
    break;
  case ElementInfo::Double:
    os << "double";
    break;
  }
  return os;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const AugmentedBundleType iface) {
  os << "iface:\n"
     << "  defName: " << iface.defName << "\n"
     << "  id: " << iface.id << "\n"
     << "  elements:\n";
  for (size_t i = 0, e = iface.elements.size(); i != e; i++) {
    auto element = iface.elements[i];
    os << "    - name: " << element.name << "\n"
       << "      id: " << element.id << "\n"
       << "      description: " << element.description << "\n"
       << "      tpe: " << element.tpe;
    if (i < e - 1)
      os << "\n";
  }
  return os;
};

// enum IDKind { Parent, Companion, Element };

// using llvm::PointerSumType;
// using llvm::PointerSumTypeMember;

// typedef PointerSumType<IDKind, PointerSumTypeMember<Parent, Operation *>,
//                        PointerSumTypeMember<Companion, Operation *>,
//                        PointerSumTypeMember<Element, Operation *>>
//     IDSum;

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

  // A map storing mutable information about an element in an interface.  This
  // is keyed using a (defName, name) tuple where defname is the name of the
  // interface and name is the name of the element.
  typedef DenseMap<std::pair<StringRef, StringRef>, ElementInfo> InterfaceMap;

private:
  // Mapping of ID to leaf graound type associated with that ID.
  llvm::DenseMap<Attribute, Value> leafMap;

  // Mapping of ID to parent instance and module.
  llvm::DenseMap<Attribute, std::pair<InstanceOp, FModuleOp>> parentIDMap;

  // Mapping of ID to companion module.
  llvm::DenseMap<Attribute, CompanionInfo> companionIDMap;

  bool unfoldField(Attribute maybeField, SmallVector<Element> &tpe,
                   IntegerAttr id, Twine path, bool buildIFace = false);

  bool unfoldBundle(Annotation anno, IntegerAttr id = {}, Twine path = {},
                    bool buildIFace = false);

  FModuleOp getEnclosingModule(Value value);

  std::variant<std::string, Type> computeType(ArrayRef<Element> elements,
                                              OpBuilder &builder);

  // Inforamtion about how the circuit should be extracted.  This will be
  // non-empty if an extraction annotation is found.
  llvm::Optional<ExtractionInfo> maybeExtractInfo = None;

  StringRef getOutputDirectory() {
    return maybeExtractInfo ? maybeExtractInfo.getValue().directory : ".";
  }

  InstancePaths *instancePathsRef;
};

class GrandCentralVisitor : public FIRRTLVisitor<GrandCentralVisitor> {
public:
  GrandCentralVisitor(
      GrandCentralPass::InterfaceMap &interfaceMap,
      llvm::StringMap<std::pair<StringRef, Operation *>> &companionMap,
      ExtractionInfo extractionInfo)
      : interfaceMap(interfaceMap), companionMap(companionMap),
        extractionInfo(extractionInfo) {}

private:
  /// Mutable store tracking each element in an interface.  This is indexed by a
  /// "defName" -> "name" tuple.
  GrandCentralPass::InterfaceMap &interfaceMap;

  llvm::StringMap<std::pair<StringRef, Operation *>> &companionMap;

  /// Helper to handle wires, registers, and nodes.
  void handleRef(Operation *op);

  /// A helper used by handleRef that can also be used to process ports.
  void handleRefLike(Operation *op, AnnotationSet &annotations,
                     FIRRTLType type);

  // Helper to handle ports of modules that may have Grand Central annotations.
  void handlePorts(Operation *op);

  // If true, then some error occurred while the visitor was running.  This
  // indicates that pass failure should occur.
  bool failed = false;

  // Information about how Grand Central collateral should be extracted.
  ExtractionInfo extractionInfo;

public:
  using FIRRTLVisitor<GrandCentralVisitor>::visitDecl;

  /// Visit FModuleOp and FExtModuleOp
  void visitModule(Operation *op);

  /// Visit ops that can make up an interface element.
  void visitDecl(RegOp op) { handleRef(op); }
  void visitDecl(RegResetOp op) { handleRef(op); }
  void visitDecl(WireOp op) { handleRef(op); }
  void visitDecl(NodeOp op) { handleRef(op); }
  void visitDecl(InstanceOp op);

  /// Process all other ops.  Error if any of these ops contain annotations that
  /// indicate it as being part of an interface.
  void visitUnhandledDecl(Operation *op);

  /// Returns true if an error condition occurred while visiting ops.
  bool hasFailed() { return failed; };
};

} // namespace

void GrandCentralVisitor::visitModule(Operation *op) {
  handlePorts(op);

  if (auto mod = dyn_cast_or_null<FModuleOp>(op)) {
    AnnotationSet annotations(op);
    if (auto anno = annotations.getAnnotation(
            "sifive.enterprise.grandcentral.ViewAnnotation")) {
      auto tpe = anno.getAs<StringAttr>("type");
      if (tpe && tpe.getValue() == "companion") {
        auto defName = anno.getAs<StringAttr>("defName");
        auto name = anno.getAs<StringAttr>("name");
        companionMap.insert_or_assign(defName.getValue(),
                                      std::make_pair(name.getValue(), op));
        auto *ctx = op->getContext();
        op->setAttr("output_file",
                    hw::OutputFileAttr::get(
                        StringAttr::get(ctx, extractionInfo.directory),
                        StringAttr::get(ctx, mod.getName() + ".sv"),
                        BoolAttr::get(ctx, true), BoolAttr::get(ctx, true),
                        ctx));
      }
    }
    for (auto &stmt : op->getRegion(0).front())
      dispatchVisitor(&stmt);
  }
}

/// Process all other operations.  This will throw an error if the operation
/// contains any annotations that indicates that this should be included in an
/// interface.  Otherwise, this is a valid nop.
void GrandCentralVisitor::visitUnhandledDecl(Operation *op) {
  AnnotationSet annotations(op);
  auto anno = annotations.getAnnotation(
      "sifive.enterprise.grandcentral.AugmentedGroundType");
  if (!anno)
    return;
  auto diag =
      op->emitOpError()
      << "is marked as a an interface element, but this op or its ports are "
         "not supposed to be interface elements (Are your annotations "
         "malformed? Is this a missing feature that should be supported?)";
  diag.attachNote()
      << "this annotation marked the op as an interface element: '" << anno
      << "'";
  failed = true;
}

/// Process annotations associated with an operation and having some type.
/// Return the annotations with the processed annotations removed.  If all
/// annotations are removed, this returns an empty ArrayAttr.
void GrandCentralVisitor::handleRefLike(mlir::Operation *op,
                                        AnnotationSet &annotations,
                                        FIRRTLType type) {
  if (annotations.empty())
    return;

  for (auto anno : annotations) {
    if (!anno.isClass("sifive.enterprise.grandcentral.AugmentedGroundType"))
      continue;

    auto defName = anno.getMember<StringAttr>("defName");
    auto name = anno.getMember<StringAttr>("name");
    if (!defName || !name) {
      op->emitOpError(
            "is marked as part of an interface, but is missing 'defName' or "
            "'name' fields (did you forget to add these?)")
              .attachNote()
          << "the full annotation is: " << anno.getDict();
      failed = true;
      continue;
    }

    // TODO: This is ignoring situations where the leaves of and interface are
    // not ground types.  This enforces the requirement that this runs after
    // LowerTypes.  However, this could eventually be relaxed.
    if (!type.isGround()) {
      auto diag = op->emitOpError()
                  << "cannot be added to interface '" << defName.getValue()
                  << "', component '" << name.getValue()
                  << "' because it is not a ground type. (Got type '" << type
                  << "'.) This will be dropped from the interface. (Did you "
                     "forget to run LowerTypes?)";
      diag.attachNote()
          << "The annotation indicating that this should be added was: '"
          << anno.getDict();
      failed = true;
      continue;
    }

    auto &component = interfaceMap[{defName.getValue(), name.getValue()}];
    component.found = true;
    component.op = op;

    switch (component.tpe) {
    case ElementInfo::Vector:
      component.width = type.getBitWidthOrSentinel();
      component.depth++;
      component.locations.push_back(op->getLoc());
      break;
    case ElementInfo::Ground:
      component.width = type.getBitWidthOrSentinel();
      component.locations.push_back(op->getLoc());
      break;
    case ElementInfo::Bundle:
    case ElementInfo::String:
    case ElementInfo::Boolean:
    case ElementInfo::Integer:
    case ElementInfo::Double:
      break;
    case ElementInfo::Error:
      llvm_unreachable("Shouldn't be here");
      break;
    }
    annotations.removeAnnotation(anno);
  }
}

/// Combined logic to handle Wires, Registers, and Nodes because these all use
/// the same approach.
void GrandCentralVisitor::handleRef(mlir::Operation *op) {
  auto annotations = AnnotationSet(op);
  handleRefLike(op, annotations, op->getResult(0).getType().cast<FIRRTLType>());
  annotations.applyToOperation(op);
}

/// Remove Grand Central Annotations from ports of modules or external modules.
/// Return argument attributes with annotations removed.
void GrandCentralVisitor::handlePorts(Operation *op) {

  SmallVector<Attribute> newArgAttrs;
  auto ports = cast<FModuleLike>(op).getPorts();
  for (size_t i = 0, e = ports.size(); i != e; ++i) {
    auto port = ports[i];
    handleRefLike(op, port.annotations, port.type);
    newArgAttrs.push_back(port.annotations.getArrayAttr());
  }

  op->setAttr("portAnnotations", ArrayAttr::get(op->getContext(), newArgAttrs));
}

void GrandCentralVisitor::visitDecl(InstanceOp op) {

  // If this instance's underlying module has a "companion" annotation, then
  // move this onto the actual instance op.
  AnnotationSet annotations(op.getReferencedModule());
  if (auto anno = annotations.getAnnotation(
          "sifive.enterprise.grandcentral.ViewAnnotation")) {
    auto tpe = anno.getAs<StringAttr>("type");
    if (!tpe) {
      op.getReferencedModule()->emitOpError(
          "contains a ViewAnnotation that does not contain a \"type\" field");
      failed = true;
      return;
    }
    if (tpe.getValue() == "companion" && !extractionInfo.bindFilename.empty()) {
      op->setAttr("lowerToBind", BoolAttr::get(op.getContext(), true));
      op->setAttr(
          "output_file",
          hw::OutputFileAttr::get(
              StringAttr::get(op.getContext(), ""),
              StringAttr::get(op.getContext(), extractionInfo.bindFilename),
              /*exclude_from_filelist=*/BoolAttr::get(op.getContext(), true),
              /*exclude_replicated_ops=*/BoolAttr::get(op.getContext(), true),
              op.getContext()));
    }
  }

  visitUnhandledDecl(op);
}

bool GrandCentralPass::unfoldField(Attribute maybeField,
                                   SmallVector<Element> &tpe, IntegerAttr id,
                                   Twine path, bool buildIFace) {
  assert(id && "id must be something inside unfoldField");

  auto field = maybeField.dyn_cast_or_null<DictionaryAttr>();
  if (!field)
    return false;
  auto clazz = field.getAs<StringAttr>("class");
  if (!clazz)
    return false;
  switch (llvm::StringSwitch<ElementInfo::Kind>(clazz.getValue())
              .Case("sifive.enterprise.grandcentral.AugmentedBundleType",
                    ElementInfo::Bundle)
              .Case("sifive.enterprise.grandcentral.AugmentedVectorType",
                    ElementInfo::Vector)
              .Case("sifive.enterprise.grandcentral.AugmentedGroundType",
                    ElementInfo::Ground)
              .Case("sifive.enterprise.grandcentral.AugmentedStringType",
                    ElementInfo::String)
              .Case("sifive.enterprise.grandcentral.AugmentedBooleanType",
                    ElementInfo::Boolean)
              .Case("sifive.enterprise.grandcentral.AugmentedIntegerType",
                    ElementInfo::Integer)
              .Case("sifive.enterprise.grandcentral.AugmentedDoubleType",
                    ElementInfo::Double)
              .Default(ElementInfo::Error)) {
  case ElementInfo::Bundle: {
    auto name = field.getAs<StringAttr>("defName");
    if (!name)
      return false;
    if (buildIFace)
      tpe.push_back(std::move(BundleKind(name.getValue())));
    return unfoldBundle(Annotation(field), id, path, buildIFace);
  }
  case ElementInfo::Vector: {
    auto name = field.getAs<StringAttr>("name");
    if (!name)
      return false;
    auto elements = field.getAs<ArrayAttr>("elements");
    if (!elements)
      return false;
    if (buildIFace)
      tpe.push_back(
          std::move(VectorKind(name.getValue(), elements.getValue().size())));
    bool failed = true;
    for (size_t i = 0, e = elements.size(); i != e; ++i) {
      failed &= unfoldField(elements[i], tpe, id, path + "[" + Twine(i) + "]",
                            (i == 0) && buildIFace);
    }
    return failed;
  }
  case ElementInfo::Ground: {
    auto name = field.getAs<StringAttr>("name");
    if (!name)
      return false;
    auto groundID = field.getAs<IntegerAttr>("id");
    if (!groundID)
      return false;

    auto builder =
        OpBuilder::atBlockEnd(companionIDMap.lookup(id).mapping.getBodyBlock());

    auto leafValue = leafMap.lookup(groundID);
    if (!leafValue)
      return false;

    auto srcPaths =
        instancePathsRef->getAbsolutePaths(getEnclosingModule(leafValue));
    assert(srcPaths.size() == 1 &&
           "Unable to handle multiply instantiated companions");
    SmallString<0> srcRef;
    for (auto path : srcPaths[0])
      srcRef.append((path.name() + ".").str());
    // srcRef.append(parentIDMap.lookup(id).first.name());

    if (auto blockArg = leafValue.dyn_cast<BlockArgument>()) {
      FModuleOp module = cast<FModuleOp>(blockArg.getOwner()->getParentOp());
      builder.create<sv::VerbatimOp>(
          getOperation().getLoc(),
          "assign " + path + " = " + srcRef +
              module.portNames()[blockArg.getArgNumber()]
                  .cast<StringAttr>()
                  .getValue() +
              ";");
    } else
      builder.create<sv::VerbatimOp>(getOperation().getLoc(),
                                     "assign " + path + " = " + srcRef +
                                         leafValue.getDefiningOp()
                                             ->getAttr("name")
                                             .cast<StringAttr>()
                                             .getValue() +
                                         ";");

    auto width = leafValue.getType().cast<FIRRTLType>().getBitWidthOrSentinel();
    if (buildIFace)
      tpe.push_back(GroundKind(name.getValue(), width));
    return true;
  }
  case ElementInfo::Error:
    return false;
  default:
    break;
  }

  // Compute the instantiation name which is "defName" for BundleType and "name"
  // for anything else.
  auto name = field.getAs<StringAttr>("name");
  if (!name)
    name = field.getAs<StringAttr>("defName");
  if (!name)
    return false;

  StringRef description = {};
  if (auto maybeDescription = field.getAs<StringAttr>("description"))
    description = maybeDescription.getValue();

  return true;
}

/// Unfold an annotation containing an AugmentedBundleType into a sequence of
/// AugmentedBundleTypes that it contains.  Returns false if the input
/// annotation was not a legal AugmentedBundleType..
bool GrandCentralPass::unfoldBundle(Annotation anno, IntegerAttr id, Twine path,
                                    bool buildIFace) {

  auto defName = anno.getMember<StringAttr>("defName");
  auto elements = anno.getMember<ArrayAttr>("elements");
  if (!defName || !elements)
    return false;
  // Set the ID if it is not already set.  Only the outermost BundleType will
  // have an ID set.
  auto isRoot = false;
  if (!id) {
    isRoot = true;
    id = anno.getMember<IntegerAttr>("id");
    if (!id)
      return false;
  }

  auto loc = getOperation().getLoc();
  auto builder = OpBuilder::atBlockEnd(getOperation().getBody());
  auto *ctx = builder.getContext();
  Operation *iface;
  if (buildIFace) {
    iface = builder.create<sv::InterfaceOp>(loc, defName.getValue());
    iface->setAttr("output_file",
                   hw::OutputFileAttr::get(
                       StringAttr::get(ctx, getOutputDirectory()),
                       StringAttr::get(ctx, defName.getValue() + ".sv"),
                       BoolAttr::get(ctx, true), BoolAttr::get(ctx, true),
                       ctx));

    // If this is the root interface, then it needs to be instantiated in the
    // parent.
    if (isRoot) {
      builder.setInsertionPointToEnd(
          parentIDMap.lookup(id).second.getBodyBlock());
      auto instance = builder.create<sv::InterfaceInstanceOp>(
          loc, cast<sv::InterfaceOp>(iface).getInterfaceType(),
          companionIDMap.lookup(id).name,
          builder.getStringAttr("__" + companionIDMap.lookup(id).name + "_" +
                                defName.getValue() + "__"));

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
                /*exclude_from_filelist=*/builder.getBoolAttr(true),
                /*exclude_replicated_ops=*/builder.getBoolAttr(true),
                bind.getContext()));
      }
    }

    builder.setInsertionPointToEnd(cast<sv::InterfaceOp>(iface).getBodyBlock());
  }

  for (auto element : elements) {
    auto field = element.dyn_cast_or_null<DictionaryAttr>();
    if (!field)
      return false;
    auto clazz = field.getAs<StringAttr>("class");
    if (!clazz)
      return false;
    auto name = field.getAs<StringAttr>("name");
    if (!name)
      return false;

    SmallVector<Element> tpe;
    auto fields = unfoldField(element, tpe, id,
                              path.isTriviallyEmpty()
                                  ? defName.getValue() + "." + name.getValue()
                                  : path + "." + name.getValue(),
                              buildIFace);
    if (!fields)
      return false;

    std::vector<Element> bar;
    for (auto a : tpe)
      bar.push_back(a);

    if (buildIFace) {
      auto description = field.getAs<StringAttr>("description");
      if (description)
        builder.create<sv::VerbatimOp>(loc,
                                       ("// " + description.getValue()).str());

      std::variant<std::string, Type> foo = computeType(bar, builder);

      if (std::holds_alternative<std::string>(foo))
        builder.create<sv::VerbatimOp>(loc, std::get<0>(foo));
      else
        builder.create<sv::InterfaceSignalOp>(
            getOperation().getLoc(), name.getValue(), std::get<1>(foo));
    }
  }

  if (buildIFace)
    llvm::errs() << *iface << "\n";

  return true;
}

FModuleOp GrandCentralPass::getEnclosingModule(Value value) {
  if (auto blockArg = value.dyn_cast<BlockArgument>())
    return cast<FModuleOp>(blockArg.getOwner()->getParentOp());

  auto *op = value.getDefiningOp();
  if (InstanceOp instance = dyn_cast<InstanceOp>(op))
    return cast<FModuleOp>(instance.getReferencedModule());

  return op->getParentOfType<FModuleOp>();
}

std::variant<std::string, Type>
GrandCentralPass::computeType(ArrayRef<Element> elements, OpBuilder &builder) {

  bool stringEmission = false;
  std::string str = "";
  llvm::raw_string_ostream s(str);
  Type tpe;
  for (auto element : llvm::reverse(elements)) {
    TypeSwitch<Element *>(&element)
        .Case<GroundKind>([&](auto a) {
          tpe = builder.getIntegerType(a->getWidth());
          return;
        })
        .Case<VectorKind>([&](auto a) {
          if (stringEmission) {
            s << "[" << a->getDepth() << "]";
            return;
          }
          tpe = hw::UnpackedArrayType::get(tpe, a->getDepth());
          return;
        })
        .Case<BundleKind>([&](auto a) {
          stringEmission = true;
          s << a->name << " " << a->name;
        });
  }

  if (stringEmission) {
    s << ";";
    return str;
  }

  return tpe;
}

void GrandCentralPass::runOnOperation() {
  CircuitOp circuitOp = getOperation();

  AnnotationSet annotations(circuitOp);
  if (annotations.empty())
    return;

  // Utility that acts like emitOpError, but does _not_ include a note.  The
  // note in emitOpError includes the entire op which means the **ENTIRE**
  // FIRRTL circuit.  This doesn't communicate anything useful to the user other
  // than flooding their terminal.
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
  // Instances link to their definitions via symbols and we don't want to break
  // this.
  auto builder = OpBuilder::atBlockEnd(circuitOp.getBody());

  SymbolTable symbolTable(circuitOp);

  removalError = false;
  circuitOp.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<RegOp, RegResetOp, WireOp, NodeOp>([&](auto op) {
          AnnotationSet annotations = AnnotationSet(op);
          annotations.removeAnnotations([&](Annotation annotation) {
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
          annotations.applyToOperation(op);
        })
        .Case<InstanceOp>([&](auto op) {
          /* Populate the leafMap for ports. */
        })
        .Case<FModuleOp>([&](FModuleOp op) {
          // Handle annotations on the ports.
          auto ports = op.getPorts();
          for (size_t i = 0, e = ports.size(); i != e; ++i) {
            auto annotations = AnnotationSet::forPort(op, i);
            annotations.removeAnnotations([&](Annotation annotation) {
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
          }

          // Handle annotations on the module.
          AnnotationSet annotations = AnnotationSet(op);
          annotations.removeAnnotations([&](Annotation annotation) {
            if (!annotation.isClass(
                    "sifive.enterprise.grandcentral.ViewAnnotation"))
              return false;
            auto tpe = annotation.getMember<StringAttr>("type");
            auto id = annotation.getMember<IntegerAttr>("id");
            auto name = annotation.getMember<StringAttr>("name");
            if (!tpe) {
              op.emitOpError()
                  << "has a malformed "
                     "'sifive.enterprise.grandcentral.ViewAnnotation' that did "
                     "not contain a 'type' field with a 'StringAttr' value";
              goto FModuleOp_error;
            }
            if (!id) {
              op.emitOpError()
                  << "has a malformed "
                     "'sifive.enterprise.grandcentral.ViewAnnotation' that did "
                     "not contain an 'id' field with an 'IntegerAttr' value";
              goto FModuleOp_error;
            }
            if (!name) {
              op.emitOpError()
                  << "has a malformed "
                     "'sifive.enterprise.grandcentral.ViewAnnotation' that did "
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
              if (!instances) {
                op.emitOpError()
                    << "is not instantiated in the design, but must be for "
                       "Grand Central to prop";
                return false;
              }

              if (std::distance(instances.getValue().begin(),
                                instances.getValue().end()) != 1) {
                auto diag = op.emitOpError()
                            << "is instantiate more than once, Grand "
                               "Central doesn't know how to handle this";
                for (auto instance : instances.getValue())
                  diag.attachNote(instance.getUser()->getLoc())
                      << "parent is instantiated here";
                return false;
              }

              auto instance =
                  cast<InstanceOp>((*(instances.getValue().begin())).getUser());

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
              if (!instances) {
                op.emitOpError()
                    << "is not instantiated in the design, but must be for "
                       "Grand Central to prop";
                return false;
              }

              if (std::distance(instances.getValue().begin(),
                                instances.getValue().end()) != 1) {
                auto diag = op.emitOpError()
                            << "is instantiate more than once, Grand "
                               "Central doesn't know how to handle this";
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
                << "has a 'sifive.enterprise.grandcentral.ViewAnnotation' with "
                   "an unknown 'type' field";
          FModuleOp_error:
            removalError = true;
            return false;
          });
        });
  });

  if (removalError)
    return signalPassFailure();

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

    AugmentedBundleType bundle;
    auto id = anno.getMember<IntegerAttr>("id");
    if (!unfoldBundle(anno, {}, companionIDMap.lookup(id).name, true)) {
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
}

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> circt::firrtl::createGrandCentralPass() {
  return std::make_unique<GrandCentralPass>();
}
