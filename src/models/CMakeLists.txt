MX_MODEL_TREE_INIT()

foreach(hdr ${MX_MODEL_HDRS_LOCAL})
  get_filename_component(_hdr_dir ${hdr} DIRECTORY)
  install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/${hdr} 
    DESTINATION 
    include/mechanica/models/${_hdr_dir}
  )
endforeach(hdr)
