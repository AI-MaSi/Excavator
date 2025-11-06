file(GLOB_RECURSE files "*.c")
add_library(sh2 ${files})
target_include_directories(sh2 INTERFACE .)