// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <gtest/gtest.h>

#include <set>

class TestInternalOrigin: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};


TEST_F(TestInternalOrigin, test)
{
    std::set<openvdb::Coord> indices;
    indices.insert(openvdb::Coord( 0,  0,  0));
    indices.insert(openvdb::Coord( 1,  0,  0));
    indices.insert(openvdb::Coord( 0,100,  8));
    indices.insert(openvdb::Coord(-9,  0,  8));
    indices.insert(openvdb::Coord(32,  0, 16));
    indices.insert(openvdb::Coord(33, -5, 16));
    indices.insert(openvdb::Coord(42,707,-35));
    indices.insert(openvdb::Coord(43, 17, 64));

    typedef openvdb::tree::Tree4<float,5,4,3>::Type FloatTree4;
    FloatTree4 tree(0.0f);
    std::set<openvdb::Coord>::iterator iter=indices.begin();
    for (int n = 0; iter != indices.end(); ++n, ++iter) {
        tree.setValue(*iter, float(1.0 + double(n) * 0.5));
    }

    openvdb::Coord C3, G;
    typedef FloatTree4::RootNodeType Node0;
    typedef Node0::ChildNodeType     Node1;
    typedef Node1::ChildNodeType     Node2;
    typedef Node2::LeafNodeType      Node3;
    for (Node0::ChildOnCIter iter0=tree.root().cbeginChildOn(); iter0; ++iter0) {//internal 1
        openvdb::Coord C0=iter0->origin();
        iter0.getCoord(G);
        EXPECT_EQ(C0,G);
        for (Node1::ChildOnCIter iter1=iter0->cbeginChildOn(); iter1; ++iter1) {//internal 2
            openvdb::Coord C1=iter1->origin();
            iter1.getCoord(G);
            EXPECT_EQ(C1,G);
            EXPECT_TRUE(C0 <= C1);
            EXPECT_TRUE(C1 <= C0 + openvdb::Coord(Node1::DIM,Node1::DIM,Node1::DIM));
            for (Node2::ChildOnCIter iter2=iter1->cbeginChildOn(); iter2; ++iter2) {//leafs
                openvdb::Coord C2=iter2->origin();
                iter2.getCoord(G);
                EXPECT_EQ(C2,G);
                EXPECT_TRUE(C1 <= C2);
                EXPECT_TRUE(C2 <= C1 + openvdb::Coord(Node2::DIM,Node2::DIM,Node2::DIM));
                for (Node3::ValueOnCIter iter3=iter2->cbeginValueOn(); iter3; ++iter3) {//leaf voxels
                    iter3.getCoord(G);
                    iter = indices.find(G);
                    EXPECT_TRUE(iter != indices.end());
                    indices.erase(iter);
                }
            }
        }
    }
    EXPECT_TRUE(indices.size() == 0);
}
