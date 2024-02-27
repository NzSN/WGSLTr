#ifndef WGSLTr_TRANSFORMER_H
#define WGSLTr_TRANSFORMER_H

#include <istream>
#include <assert.h>

#include "Types.h"

#include "Api/Transformer.h"

namespace WGSLTr {

class Transformer {
public:
  Transformer(std::initializer_list<Types::Rule> rules):
    input_{nullptr}, tr_{rules} {}

  void Setup(std::istream* s) { input_ = s; }
  void Parse() {
    assert((false && "Transformer::Parse() is not implement"));
  }

  std::optional<std::string> Transform() {
    return tr_(input_);
  }
private:
  std::istream* input_;
  Chameleon::api::Transformer<GET_LANG_TYPE(WGSL)> tr_;
};


} // WGSLTr

#endif /* WGSLTr_TRANSFORMER_H */
