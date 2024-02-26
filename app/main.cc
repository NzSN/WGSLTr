#include "argparse/argparse.hpp"
#include <fstream>

#include "WGSLTr.h"
#include "Config.h"

#include "Transformer.h"

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("WGSLTr", WGSLTr_VERSION);

  // Commandline register
  program.add_argument("-i", "--input")
    .required()
    .help("The WGSL source file to be transformed");

  program.add_argument("-o", "--output")
    .required()
    .help("The WGSL source file already transformed");

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  std::ifstream input(program.get("--input"));

  return 0;
}
