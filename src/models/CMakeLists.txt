MX_MODEL_TREE_INIT()

foreach(hdr ${MX_MODEL_HDRS_LOCAL})
  install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/${hdr} 
    DESTINATION 
    include/mechanica/models/${hdr}
  )
endforeach(hdr)
