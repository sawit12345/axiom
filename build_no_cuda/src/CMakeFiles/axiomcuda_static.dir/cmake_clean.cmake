file(REMOVE_RECURSE
  "libaxiomcuda.a"
  "libaxiomcuda.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/axiomcuda_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
