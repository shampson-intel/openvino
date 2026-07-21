// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Regression test for a use-after-free from iterator invalidation in
// Plugin::import_model_npuw() (src/plugins/intel_npu/src/plugin/src/plugin.cpp).
//
// The buggy code erased map entries by key while advancing a separate loop
// iterator over the same container:
//
//   for (auto it = properties.begin(); it != properties.end(); ++it) {
//       if (it->first.find("NPUW") != it->first.npos) {
//           properties.erase(it->first);   // invalidates `it` if it pointed at
//                                           // the erased node
//       }
//   }
//   // then ++it runs on a possibly-dangling iterator -> UB / use-after-free
//
// The fix extracts the loop into intel_npu::utils::drop_npuw_properties()
// (src/plugins/intel_npu/src/utils/include/intel_npu/utils/utils.hpp), using
// the standard erase-returns-next-iterator idiom, and import_model_npuw() now
// calls that helper directly. Extracting it also makes the exact production
// code path linkable and callable from this unit test binary.

#include "intel_npu/utils/utils.hpp"

#include <gtest/gtest.h>

TEST(DropNpuwPropertiesTest, EmptyMap) {
    ov::AnyMap properties;
    intel_npu::utils::drop_npuw_properties(properties);
    EXPECT_TRUE(properties.empty());
}

TEST(DropNpuwPropertiesTest, NoNpuwKeys) {
    ov::AnyMap properties{{"PERFORMANCE_HINT", "LATENCY"}, {"NUM_STREAMS", "1"}};
    intel_npu::utils::drop_npuw_properties(properties);
    EXPECT_EQ(properties.size(), 2u);
    EXPECT_TRUE(properties.count("PERFORMANCE_HINT"));
    EXPECT_TRUE(properties.count("NUM_STREAMS"));
}

TEST(DropNpuwPropertiesTest, AllNpuwKeys) {
    ov::AnyMap properties{{"NPUW_DEVICES", "CPU"}, {"NPUW_FOLD", "YES"}};
    intel_npu::utils::drop_npuw_properties(properties);
    EXPECT_TRUE(properties.empty());
}

TEST(DropNpuwPropertiesTest, NpuwKeyFirst) {
    // std::map keeps keys sorted; "NPUW_*" sorts before "PERFORMANCE_HINT" alphabetically.
    ov::AnyMap properties{{"NPUW_DEVICES", "CPU"}, {"PERFORMANCE_HINT", "LATENCY"}};
    intel_npu::utils::drop_npuw_properties(properties);
    ASSERT_EQ(properties.size(), 1u);
    EXPECT_TRUE(properties.count("PERFORMANCE_HINT"));
}

TEST(DropNpuwPropertiesTest, NpuwKeyLast) {
    // "PERFORMANCE_HINT" sorts before "NPUW_*"; the erased node is the last element,
    // the exact position that made the old code's ++it run past end() undefined.
    ov::AnyMap properties{{"PERFORMANCE_HINT", "LATENCY"}, {"NPUW_DEVICES", "CPU"}};
    intel_npu::utils::drop_npuw_properties(properties);
    ASSERT_EQ(properties.size(), 1u);
    EXPECT_TRUE(properties.count("PERFORMANCE_HINT"));
}

TEST(DropNpuwPropertiesTest, ConsecutiveNpuwKeys) {
    // Multiple adjacent NPUW_* entries: each erase must still yield a valid
    // iterator for the very next comparison/increment.
    ov::AnyMap properties{{"NPUW_CWAI", "YES"},
                           {"NPUW_DEVICES", "CPU"},
                           {"NPUW_FOLD", "YES"},
                           {"PERFORMANCE_HINT", "LATENCY"}};
    intel_npu::utils::drop_npuw_properties(properties);
    ASSERT_EQ(properties.size(), 1u);
    EXPECT_TRUE(properties.count("PERFORMANCE_HINT"));
}

TEST(DropNpuwPropertiesTest, InterleavedNpuwAndNonNpuwKeys) {
    ov::AnyMap properties{{"CACHE_MODE", "OPTIMIZE_SPEED"},
                           {"NPUW_CWAI", "YES"},
                           {"NUM_STREAMS", "1"},
                           {"NPUW_FOLD", "YES"},
                           {"PERFORMANCE_HINT", "LATENCY"}};
    intel_npu::utils::drop_npuw_properties(properties);
    ASSERT_EQ(properties.size(), 3u);
    EXPECT_TRUE(properties.count("CACHE_MODE"));
    EXPECT_TRUE(properties.count("NUM_STREAMS"));
    EXPECT_TRUE(properties.count("PERFORMANCE_HINT"));
    EXPECT_FALSE(properties.count("NPUW_CWAI"));
    EXPECT_FALSE(properties.count("NPUW_FOLD"));
}
