#include <gtest/gtest.h>
#include "WGSLTr.h"

#include <sstream>

namespace WGSLTr {

TEST(WGSLTrTests, Basic) {
  WGSLTrCompiler compiler;

  std::istringstream s{"fn main() { a = b + c; }"};

  // Setup Phase

  // Setup targetsource that expect to
  // parse.
  compiler.Setup(&s);

  // TODO: Implement WGSLTrCompiler::Parse()
  // compiler.Parse();

  // Analysis Phase
  // Information of ParseTree.
  // TODO: Need to implement.
  // auto t = compiler.targetTree();
  // Analyzer analyzer;
  // auto info = analyzer(t);

  // compiler.setAnalyzedInfo(info);

  // Transform Phase
  // Finally, we get outpout tree.
  auto t_transed = compiler.Compile();

  std::cout << t_transed.value() << std::endl;
}


} // WGSLTr
