# For efficiently aggregating source from lots of models

# This is for initializing processing in a subdirectory
macro(MX_MODEL_TREE_INIT )

  set(MX_MODEL_SRCS_LOCAL )
  set(MX_MODEL_HDRS_LOCAL )

endmacro(MX_MODEL_TREE_INIT)

# This is for posting a source in a subdirectory; path must be absolute or relative to this directory
macro(MX_MODEL_TREE_SRC src_path)

  list(APPEND MX_MODEL_SRCS_LOCAL ${src_path})

endmacro(MX_MODEL_TREE_SRC)

# This is for posting a header in a subdirectory; path must be relative to this directory
macro(MX_MODEL_TREE_HDR hdr_path)

  list(APPEND MX_MODEL_HDRS_LOCAL ${hdr_path})

endmacro(MX_MODEL_TREE_HDR)

# This is for incorporating info from a subdirectory; path must be w.r.t. current source directory
macro(MX_MODEL_TREE_PROC subdir)

  get_directory_property(_TMP 
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}
    DEFINITION MX_MODEL_SRCS_LOCAL
  )
  MX_MODEL_TREE_SRC(${_TMP})

  get_directory_property(_TMP 
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}
    DEFINITION MX_MODEL_HDRS_LOCAL
  )
  MX_MODEL_TREE_HDR(${_TMP})

endmacro(MX_MODEL_TREE_PROC)
