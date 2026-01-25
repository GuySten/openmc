set(TEST_NAMES
  test_distribution
  test_file_utils
  test_tally
  test_interpolate
  test_math
  test_mcpl_stat_sum
  test_mesh
  test_region
  test_deuteron
  test_charged_particle
  test_dt_fusion
  # Add additional unit test files here
)

foreach(test ${TEST_NAMES})
  add_executable(${test} ${test}.cpp)
  target_link_libraries(${test} Catch2::Catch2WithMain libopenmc)
  add_test(NAME ${test} COMMAND ${test} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endforeach()

# Copy test data files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
