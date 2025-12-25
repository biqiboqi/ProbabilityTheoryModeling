enable_testing()

add_executable(
        ${PROJECT_NAME}_tests
        main_test.cpp
        sigma_algebra_tests.cpp
        test_suites/ProjectIntegrationTestSuite.cpp
        distributions_tests.cpp
        markov_chain_tests.cpp
        law_of_large_numbers_tests.cpp
)

target_link_libraries(
        ${PROJECT_NAME}_tests
        sigma-algebra
        law-of-large-numbers
        GTest::gtest_main
)

message(STATUS "Tests build type: ${CMAKE_BUILD_TYPE}")

target_include_directories(${PROJECT_NAME}_tests PUBLIC ${PROJECT_SOURCE_DIR})

include(GoogleTest)

gtest_discover_tests(${PROJECT_NAME}_tests)
