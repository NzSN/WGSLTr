#ifndef TYPES_H
#define TYPES_H

#include "antlr4-runtime.h"
#include "Api/Rule.h"
#include "Api/Types.h"

namespace WGSLTr {
namespace Types {

using ParseTree = Chameleon::api::Types::AntlrParseTree;
using Rule = Chameleon::api::Rule<GET_LANG_TYPE(WGSL)>;

} // Types
} // WGSLTr


#endif /* TYPES_H */
