#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <istream>
#include "Types.h"



namespace WGSLTr {
namespace Transformer {

class Transformer {
public:
  Transformer(): tree_{nullptr} {}

  void Setup(std::istream* s, );
  Types::ParseTree* Transform();
private:
  Types::ParseTree* tree_;
};


} // Transformer
} // WGSLTr


#endif /* TRANSFORMER_H */
