#pragma once

#include <cstddef>

namespace yolo {

struct Size {
   size_t height = 0;
   size_t width = 0;

   Size() {}

   Size(size_t aHeight, size_t aWidth) : height(aHeight), width(aWidth) {}

   Size operator+(size_t v) {
       return Size(height + v, width + v);
   }

   Size operator-(size_t v) {
       return Size(height - v, width - v);
   }

   Size operator*(size_t v) {
       return Size(height * v, width * v);
   }

   Size operator/(size_t v) {
       return Size(height / v, width / v);
   }
};

} // namespace yolo
