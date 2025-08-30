# this also pulls ducc and adds the ducc0 target
CPMAddPackage(
    NAME finfufft
    GITHUB_REPOSITORY iboB/finufft
    VERSION 2.4.2-ibob
    GIT_TAG e41e37c
    OPTIONS
        "FINUFFT_USE_DUCC0 ON"
)

set(tgt finufft-vs-ducc)
add_executable(${tgt} ${tgt}.cpp)
target_link_libraries(${tgt} PRIVATE
    splat::splat
    finufft
    ducc0
)

# required by finufft config
set_target_properties(
    ${tgt}
    PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)
