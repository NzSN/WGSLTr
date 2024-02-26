#include <gtest/gtest.h>
#include "WGSLTr.h"

#include <fstream>

TEST(WGSLTrTests, Basic) {
  WGSLTr compiler;

  std::ifstream s{"..."};

  // Setup Phase

  // Setup targetsource that expect to
  // parse.
  compiler.setup(s);
  compiler.parse();

  // Analysis Phase
  // Information of ParseTree.
  auto t = compiler.targetTree();
  Analyzer analyzer;
  auto info = analyzer(t);

  compiler.setAnalyzedInfo(info);

  // Transform Phase
  // Finally, we get outpout tree.
  auto t_transed = compiler.compile();
}
