//===- TestPasses.cpp - Test passes for the scheduling infrastructure -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements test passes for scheduling problems and algorithms.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace circt;
using namespace circt::scheduling;

//===----------------------------------------------------------------------===//
// Construction helper methods
//===----------------------------------------------------------------------===//

static SmallVector<SmallVector<unsigned>> parseArrayOfArrays(ArrayAttr attr) {
  SmallVector<SmallVector<unsigned>> result;
  for (auto elemArr : attr.getAsRange<ArrayAttr>()) {
    result.emplace_back();
    for (auto elem : elemArr.getAsRange<IntegerAttr>())
      result.back().push_back(elem.getInt());
  }
  return result;
}

static SmallVector<std::pair<llvm::StringRef, unsigned>>
parseArrayOfDicts(ArrayAttr attr, StringRef key) {
  SmallVector<std::pair<llvm::StringRef, unsigned>> result;
  for (auto dictAttr : attr.getAsRange<DictionaryAttr>()) {
    auto name = dictAttr.getAs<StringAttr>("name");
    auto value = dictAttr.getAs<IntegerAttr>(key);
    if (name && value)
      result.push_back(std::make_pair(name.getValue(), value.getInt()));
  }
  return result;
}

static void constructProblem(Problem &prob, FuncOp func) {
  // set up catch-all operator type with unit latency
  auto unitOpr = prob.getOrInsertOperatorType("unit");
  prob.setLatency(unitOpr, 1);

  // parse additional operator type information attached to the test case
  if (auto attr = func->getAttrOfType<ArrayAttr>("operatortypes")) {
    for (auto &elem : parseArrayOfDicts(attr, "latency")) {
      auto opr = prob.getOrInsertOperatorType(std::get<0>(elem));
      prob.setLatency(opr, std::get<1>(elem));
    }
  }

  // construct problem (consider only the first block)
  for (auto &op : func.getBlocks().front().getOperations()) {
    prob.insertOperation(&op);

    if (auto oprRefAttr = op.getAttrOfType<StringAttr>("opr")) {
      auto opr = prob.getOrInsertOperatorType(oprRefAttr.getValue());
      prob.setLinkedOperatorType(&op, opr);
    } else {
      prob.setLinkedOperatorType(&op, unitOpr);
    }
  }

  // parse auxiliary dependences in the testcase, encoded as an array of arrays
  // of integer attributes
  if (auto attr = func->getAttrOfType<ArrayAttr>("auxdeps")) {
    auto &ops = prob.getOperations();
    unsigned nOps = ops.size();
    for (auto &elemArr : parseArrayOfArrays(attr)) {
      assert(elemArr.size() >= 2 && elemArr[0] < nOps && elemArr[1] < nOps);
      Operation *from = ops[elemArr[0]];
      Operation *to = ops[elemArr[1]];
      auto res = prob.insertDependence(std::make_pair(from, to));
      assert(succeeded(res));
    }
  }
}

static void constructCyclicProblem(CyclicProblem &prob, FuncOp func) {
  constructProblem(prob, func);

  // parse auxiliary dependences in the testcase (again), in order to set the
  // optional distance in the cyclic problem
  if (auto attr = func->getAttrOfType<ArrayAttr>("auxdeps")) {
    auto &ops = prob.getOperations();
    for (auto &elemArr : parseArrayOfArrays(attr)) {
      if (elemArr.size() < 3)
        continue; // skip this dependence, rather than setting the default value
      Operation *from = ops[elemArr[0]];
      Operation *to = ops[elemArr[1]];
      unsigned dist = elemArr[2];
      prob.setDistance(std::make_pair(from, to), dist);
    }
  }
}

static void constructSPOProblem(SharedPipelinedOperatorsProblem &prob,
                                FuncOp func) {
  constructProblem(prob, func);

  // parse operator type info (again) to extract optional operator limit
  if (auto attr = func->getAttrOfType<ArrayAttr>("operatortypes")) {
    for (auto &elem : parseArrayOfDicts(attr, "limit")) {
      auto opr = prob.getOrInsertOperatorType(std::get<0>(elem));
      prob.setLimit(opr, std::get<1>(elem));
    }
  }
}

static void emitSchedule(Problem &prob, StringRef attrName,
                         OpBuilder &builder) {
  for (auto *op : prob.getOperations()) {
    unsigned startTime = *prob.getStartTime(op);
    op->setAttr(attrName, builder.getI32IntegerAttr(startTime));
  }
}

//===----------------------------------------------------------------------===//
// (Basic) Problem
//===----------------------------------------------------------------------===//

namespace {
struct TestProblemPass : public PassWrapper<TestProblemPass, FunctionPass> {
  void runOnFunction() override;
};
} // namespace

void TestProblemPass::runOnFunction() {
  auto func = getFunction();

  Problem prob(func);
  constructProblem(prob, func);

  if (failed(prob.check())) {
    func->emitError("problem check failed");
    return signalPassFailure();
  }

  // get schedule from the test case
  for (auto *op : prob.getOperations())
    if (auto startTimeAttr = op->getAttrOfType<IntegerAttr>("problemStartTime"))
      prob.setStartTime(op, startTimeAttr.getInt());

  if (failed(prob.verify())) {
    func->emitError("problem verification failed");
    return signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// CyclicProblem
//===----------------------------------------------------------------------===//

namespace {
struct TestCyclicProblemPass
    : public PassWrapper<TestCyclicProblemPass, FunctionPass> {
  void runOnFunction() override;
};
} // namespace

void TestCyclicProblemPass::runOnFunction() {
  auto func = getFunction();

  CyclicProblem prob(func);
  constructCyclicProblem(prob, func);

  if (failed(prob.check())) {
    func->emitError("problem check failed");
    return signalPassFailure();
  }

  // get II from the test case
  if (auto attr = func->getAttrOfType<IntegerAttr>("problemInitiationInterval"))
    prob.setInitiationInterval(attr.getInt());

  // get schedule from the test case
  for (auto *op : prob.getOperations())
    if (auto startTimeAttr = op->getAttrOfType<IntegerAttr>("problemStartTime"))
      prob.setStartTime(op, startTimeAttr.getInt());

  if (failed(prob.verify())) {
    func->emitError("problem verification failed");
    return signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// SharedPipelinedOperatorsProblem
//===----------------------------------------------------------------------===//

namespace {
struct TestSPOProblemPass
    : public PassWrapper<TestSPOProblemPass, FunctionPass> {
  void runOnFunction() override;
};
} // namespace

void TestSPOProblemPass::runOnFunction() {
  auto func = getFunction();

  SharedPipelinedOperatorsProblem prob(func);
  constructSPOProblem(prob, func);

  if (failed(prob.check())) {
    func->emitError("problem check failed");
    return signalPassFailure();
  }

  // get schedule from the test case
  for (auto *op : prob.getOperations())
    if (auto startTimeAttr = op->getAttrOfType<IntegerAttr>("problemStartTime"))
      prob.setStartTime(op, startTimeAttr.getInt());

  if (failed(prob.verify())) {
    func->emitError("problem verification failed");
    return signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// ASAPScheduler
//===----------------------------------------------------------------------===//

namespace {
struct TestASAPSchedulerPass
    : public PassWrapper<TestASAPSchedulerPass, FunctionPass> {
  void runOnFunction() override;
};
} // anonymous namespace

void TestASAPSchedulerPass::runOnFunction() {
  auto func = getFunction();

  Problem prob(func);
  constructProblem(prob, func);
  assert(succeeded(prob.check()));

  if (failed(scheduleASAP(prob))) {
    func->emitError("scheduling failed");
    return signalPassFailure();
  }

  if (failed(prob.verify())) {
    func->emitError("schedule verification failed");
    return signalPassFailure();
  }

  OpBuilder builder(func.getContext());
  emitSchedule(prob, "asapStartTime", builder);
}

//===----------------------------------------------------------------------===//
// SimplexScheduler
//===----------------------------------------------------------------------===//

namespace {
struct TestSimplexSchedulerPass
    : public PassWrapper<TestSimplexSchedulerPass, FunctionPass> {
  TestSimplexSchedulerPass() = default;
  TestSimplexSchedulerPass(const TestSimplexSchedulerPass &) {}
  Option<std::string> problemToTest{*this, "with", llvm::cl::init("Problem")};
  void runOnFunction() override;
};
} // anonymous namespace

void TestSimplexSchedulerPass::runOnFunction() {
  auto func = getFunction();
  Operation *lastOp = func.getBlocks().front().getTerminator();
  OpBuilder builder(func.getContext());

  if (problemToTest == "Problem") {
    Problem prob(func);
    constructProblem(prob, func);
    assert(succeeded(prob.check()));

    if (failed(scheduleSimplex(prob, lastOp))) {
      func->emitError("scheduling failed");
      return signalPassFailure();
    }

    if (failed(prob.verify())) {
      func->emitError("schedule verification failed");
      return signalPassFailure();
    }

    emitSchedule(prob, "simplexStartTime", builder);
    return;
  }

  if (problemToTest == "CyclicProblem") {
    CyclicProblem prob(func);
    constructCyclicProblem(prob, func);
    assert(succeeded(prob.check()));

    if (failed(scheduleSimplex(prob, lastOp))) {
      func->emitError("scheduling failed");
      return signalPassFailure();
    }

    if (failed(prob.verify())) {
      func->emitError("schedule verification failed");
      return signalPassFailure();
    }

    func->setAttr("simplexInitiationInterval",
                  builder.getI32IntegerAttr(*prob.getInitiationInterval()));
    emitSchedule(prob, "simplexStartTime", builder);
    return;
  }

  if (problemToTest == "SharedPipelinedOperatorsProblem") {
    SharedPipelinedOperatorsProblem prob(func);
    constructProblem(prob, func);
    constructSPOProblem(prob, func);
    assert(succeeded(prob.check()));

    if (failed(scheduleSimplex(prob, lastOp))) {
      func->emitError("scheduling failed");
      return signalPassFailure();
    }

    if (failed(prob.verify())) {
      func->emitError("schedule verification failed");
      return signalPassFailure();
    }

    emitSchedule(prob, "simplexStartTime", builder);
    return;
  }

  llvm_unreachable("Unsupported scheduling problem");
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace circt {
namespace test {
void registerSchedulingTestPasses() {
  PassRegistration<TestProblemPass> problemTester(
      "test-scheduling-problem", "Import a schedule encoded as attributes");
  PassRegistration<TestCyclicProblemPass> cyclicProblemTester(
      "test-cyclic-problem",
      "Import a solution for the cyclic problem encoded as attributes");
  PassRegistration<TestSPOProblemPass> spoTester(
      "test-spo-problem", "Import a solution for the shared, pipelined "
                          "operators problem encoded as attributes");
  PassRegistration<TestASAPSchedulerPass> asapTester(
      "test-asap-scheduler", "Emit ASAP scheduler's solution as attributes");
  PassRegistration<TestSimplexSchedulerPass> simplexTester(
      "test-simplex-scheduler",
      "Emit a simplex scheduler's solution as attributes");
}
} // namespace test
} // namespace circt
