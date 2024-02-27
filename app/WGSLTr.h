#ifndef WGSLTR_H
#define WGSLTR_H

#include <istream>
#include <optional>
#include <assert.h>

#include "Transformer.h"

namespace WGSLTr {

class WGSLTrCompiler {
public:
  WGSLTrCompiler():
    tr_{
      {"__a + __b", "__b + __b"},
      {"__b + __c", "__c * __c"},
    } {}

  void Setup(std::istream* s) {
    tr_.Setup(s);
  }

  void Parse() {
    assert((false && "WGSLTrCompiler::Parse() is not implement"));
  }

  void Analysis() {
    assert((false && "Analyzer is not implemented."));
  }

  std::optional<std::string>
  Compile() {
    return tr_.Transform();
  }
private:
  Transformer tr_;
};

} // WGSLTr


#endif /* WGSLTR_H */
