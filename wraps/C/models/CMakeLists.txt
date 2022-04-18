MX_MODEL_TREE_INIT()

add_subdirectory(center)
MX_MODEL_TREE_PROC(center)

foreach(hdr ${MX_MODEL_HDRS_LOCAL})
  get_filename_component(_hdr_dir ${hdr} DIRECTORY)
  install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/${hdr} 
    DESTINATION 
    ${MX_WRAPS_C_INSTALL_PREFIX}/models/${_hdr_dir}
  )
endforeach(hdr)
