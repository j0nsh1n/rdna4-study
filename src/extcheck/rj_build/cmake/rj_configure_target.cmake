# Stand-in for rocjitsu's own helper (not in the download): only what rj_arith needs to compile.
function(rj_configure_target target)
    set_target_properties(${target} PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
endfunction()
