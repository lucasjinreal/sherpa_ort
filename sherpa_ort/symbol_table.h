#pragma once

#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace sherpa_onnx {

class SymbolTable {
public:
  SymbolTable() = default;

  explicit SymbolTable(const std::string &filename) {
    std::ifstream is(filename);
    std::string sym;
    int32_t id;
    while (is >> sym >> id) {
      if (sym.size() >= 3) {
        // For BPE-based models, we replace ▁ with a space
        // Unicode 9601, hex 0x2581, utf8 0xe29681
        const uint8_t *p = reinterpret_cast<const uint8_t *>(sym.c_str());
        if (p[0] == 0xe2 && p[1] == 0x96 && p[2] == 0x81) {
          sym = sym.replace(0, 3, " ");
        }
      }

      assert(!sym.empty());
      assert(sym2id_.count(sym) == 0);
      assert(id2sym_.count(id) == 0);

      sym2id_.insert({sym, id});
      id2sym_.insert({id, sym});
    }
    assert(is.eof());
  }

  std::string ToString() const {
    std::ostringstream os;
    char sep = ' ';
    for (const auto &p : sym2id_) {
      os << p.first << sep << p.second << "\n";
    }
    return os.str();
  }

  const std::string &operator[](int32_t id) const { return id2sym_.at(id); }
  int32_t operator[](const std::string &sym) const { return sym2id_.at(sym); }

  bool contains(int32_t id) const { return id2sym_.count(id) != 0; }

  bool contains(const std::string &sym) const {
    return sym2id_.count(sym) != 0;
  }

private:
  std::unordered_map<std::string, int32_t> sym2id_;
  std::unordered_map<int32_t, std::string> id2sym_;
};

inline std::ostream &operator<<(std::ostream &os, const SymbolTable &symbol_table) {
  return os << symbol_table.ToString();
}
} // namespace sherpa_onnx
