// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/TypeList.h>

#include <gtest/gtest.h>

#include <functional> // for std::ref()
#include <string>


using namespace openvdb;

class TestTypeList : public ::testing::Test
{
};

////////////////////////////////////////


namespace {

template<typename T> char typeCode() { return '.'; }
template<> char typeCode<bool>()   { return 'b'; }
template<> char typeCode<char>()   { return 'c'; }
template<> char typeCode<double>() { return 'd'; }
template<> char typeCode<float>()  { return 'f'; }
template<> char typeCode<int>()    { return 'i'; }
template<> char typeCode<long>()   { return 'l'; }


struct TypeCodeOp
{
    std::string codes;
    template<typename T> void operator()(const T&) { codes.push_back(typeCode<T>()); }
};

struct ListModifier
{
    template <typename T>
    using Promote = typename PromoteType<T>::Next;

    template <typename T>
    using RemoveInt32 = typename std::conditional<std::is_same<int32_t, T>::value, TypeList<>, T>::type;

    template <typename T>
    using Duplicate = TypeList<T, T>;
};

template<typename TSet>
inline std::string
typeSetAsString()
{
    TypeCodeOp op;
    TSet::foreach(std::ref(op));
    return op.codes;
}

template <typename T1, typename T2>
using ConvertIntegrals = typename std::conditional<std::is_integral<T1>::value, T2, T1>::type;

template <typename T1>
using ConvertIntegralsToFloats = ConvertIntegrals<T1, float>;

template <typename T>
struct Tester
{
    template <typename T1>
    using ConvertIntegralsToFloats = ConvertIntegrals<T1, T>;
};

} // anonymous namespace

TEST_F(TestTypeList, testTypeList)
{
    /// Compile time tests of TypeList

    using IntTypes = TypeList<Int16, Int32, Int64>;
    using EmptyList = TypeList<>;

    // To TupleList
    static_assert(std::is_same<EmptyList::AsTupleList, TupleList<>>::value);
    static_assert(std::is_same<IntTypes::AsTupleList, TupleList<Int16, Int32, Int64>>::value);

    // Size
    static_assert((IntTypes::Size == 3));
    static_assert((EmptyList::Size == 0));

    // Contains
    static_assert((IntTypes::Contains<Int16>));
    static_assert((IntTypes::Contains<Int32>));
    static_assert((IntTypes::Contains<Int64>));
    static_assert((!IntTypes::Contains<float>));

    // Index
    static_assert((IntTypes::Index<Int16> == 0));
    static_assert((IntTypes::Index<Int32> == 1));
    static_assert((IntTypes::Index<Int64> == 2));
    static_assert((IntTypes::Index<float> == -1));

    // Get
    static_assert((std::is_same<IntTypes::Get<0>, Int16>::value));
    static_assert((std::is_same<IntTypes::Get<1>, Int32>::value));
    static_assert((std::is_same<IntTypes::Get<2>, Int64>::value));
    static_assert((std::is_same<IntTypes::Get<3>,  typelist_internal::NullType>::value));
    static_assert((!std::is_same<IntTypes::Get<3>, void>::value));

    // Unique
    static_assert((std::is_same<IntTypes::Unique<>, IntTypes>::value));
    static_assert((std::is_same<IntTypes::Unique<IntTypes>, IntTypes>::value));
    static_assert((std::is_same<EmptyList::Unique<>, EmptyList>::value));
    static_assert((std::is_same<TypeList<int, int, int>::Unique<>, TypeList<int>>::value));
    static_assert((std::is_same<TypeList<float, int, float>::Unique<>, TypeList<float, int>>::value));
    static_assert((std::is_same<TypeList<bool, int, float, int, float>::Unique<>, TypeList<bool, int, float>>::value));

    // Front/Back
    static_assert((std::is_same<IntTypes::Front, Int16>::value));
    static_assert((std::is_same<IntTypes::Back, Int64>::value));

    // PopFront/PopBack
    static_assert((std::is_same<IntTypes::PopFront, TypeList<Int32, Int64>>::value));
    static_assert((std::is_same<IntTypes::PopBack, TypeList<Int16, Int32>>::value));

    // RemoveByIndex
    static_assert((std::is_same<IntTypes::RemoveByIndex<0,0>, IntTypes::PopFront>::value));
    static_assert((std::is_same<IntTypes::RemoveByIndex<2,2>, IntTypes::PopBack>::value));
    static_assert((std::is_same<IntTypes::RemoveByIndex<0,2>, EmptyList>::value));
    static_assert((std::is_same<IntTypes::RemoveByIndex<1,2>, TypeList<Int16>>::value));
    static_assert((std::is_same<IntTypes::RemoveByIndex<1,1>, TypeList<Int16, Int64>>::value));
    static_assert((std::is_same<IntTypes::RemoveByIndex<0,1>, TypeList<Int64>>::value));
    static_assert((std::is_same<IntTypes::RemoveByIndex<0,10>, EmptyList>::value));

    // invalid indices do nothing
    static_assert((std::is_same<IntTypes::RemoveByIndex<2,1>, IntTypes>::value));
    static_assert((std::is_same<IntTypes::RemoveByIndex<3,3>, IntTypes>::value));

    //

    // Test methods on an empty list
    static_assert((!EmptyList::Contains<Int16>));
    static_assert((EmptyList::Index<Int16> == -1));
    static_assert((std::is_same<EmptyList::Get<0>, typelist_internal::NullType>::value));
    static_assert((std::is_same<EmptyList::Front, typelist_internal::NullType>::value));
    static_assert((std::is_same<EmptyList::Back, typelist_internal::NullType>::value));
    static_assert((std::is_same<EmptyList::PopFront, EmptyList>::value));
    static_assert((std::is_same<EmptyList::PopBack, EmptyList>::value));
    static_assert((std::is_same<EmptyList::RemoveByIndex<0,0>, EmptyList>::value));

    //

    // Test some methods on lists with duplicate types
    using DuplicateIntTypes = TypeList<Int32, Int16, Int64, Int16>;
    using DuplicateRealTypes = TypeList<float, float, float, float>;

    static_assert((DuplicateIntTypes::Size == 4));
    static_assert((DuplicateRealTypes::Size == 4));
    static_assert((DuplicateIntTypes::Index<Int16> == 1));
    static_assert((std::is_same<DuplicateIntTypes::Unique<>, TypeList<Int32, Int16, Int64>>::value));
    static_assert((std::is_same<DuplicateRealTypes::Unique<>, TypeList<float>>::value));
    static_assert((std::is_same<DuplicateRealTypes::Unique<DuplicateIntTypes>,
        TypeList<float, Int32, Int16, Int64>>::value));

    //

    // Tests on VDB grid node chains - reverse node chains from leaf->root
    using Tree4Float = openvdb::tree::Tree4<float, 5, 4, 3>::Type; // usually the same as FloatTree
    using NodeChainT = Tree4Float::RootNodeType::NodeChainType;

    // Expected types
    using LeafT = openvdb::tree::LeafNode<float, 3>;
    using IternalT1 = openvdb::tree::InternalNode<LeafT, 4>;
    using IternalT2 = openvdb::tree::InternalNode<IternalT1, 5>;
    using RootT = openvdb::tree::RootNode<IternalT2>;

    static_assert((std::is_same<NodeChainT::Get<0>, LeafT>::value));
    static_assert((std::is_same<NodeChainT::Get<1>, IternalT1>::value));
    static_assert((std::is_same<NodeChainT::Get<2>, IternalT2>::value));
    static_assert((std::is_same<NodeChainT::Get<3>, RootT>::value));
    static_assert((std::is_same<NodeChainT::Get<4>, typelist_internal::NullType>::value));

    // Transform
    static_assert((std::is_same<EmptyList::Transform<ListModifier::Promote>, EmptyList>::value));
    static_assert((std::is_same<TypeList<int32_t>::Transform<ListModifier::Promote>, TypeList<int64_t>>::value));
    static_assert((std::is_same<TypeList<float>::Transform<ListModifier::Promote>, TypeList<double>>::value));
    static_assert((std::is_same<TypeList<bool, uint32_t>::Transform<ListModifier::Promote>, TypeList<uint16_t, uint64_t>>::value));

    static_assert((std::is_same<TypeList<float, uint32_t>::Transform<ConvertIntegralsToFloats>,
        TypeList<float, float>>::value));
    static_assert((std::is_same<TypeList<float, uint32_t>::Transform<Tester<float>::ConvertIntegralsToFloats>,
        TypeList<float, float>>::value));

    // @note  Transforming types to TypeList<>s causes the target type to expand.
    //   This has some weird effects like being able to remove/extend types with
    //   TypeList::Transform.
    static_assert((std::is_same<TypeList<int32_t, int32_t>::Transform<ListModifier::RemoveInt32>, EmptyList>::value));
    static_assert((std::is_same<TypeList<int32_t, float>::Transform<ListModifier::Duplicate>,
        TypeList<int32_t, int32_t, float, float>>::value));

    // Foreach
    using T0 = TypeList<>;
    EXPECT_EQ(std::string(), typeSetAsString<T0>());

    using T1 = TypeList<int>;
    EXPECT_EQ(std::string("i"), typeSetAsString<T1>());

    using T2 = TypeList<float>;
    EXPECT_EQ(std::string("f"), typeSetAsString<T2>());

    using T3 = TypeList<bool, double>;
    EXPECT_EQ(std::string("bd"), typeSetAsString<T3>());

    using T4 = T1::Append<T2>;
    EXPECT_EQ(std::string("if"), typeSetAsString<T4>());
    EXPECT_EQ(std::string("fi"), typeSetAsString<T2::Append<T1>>());

    using T5 = T3::Append<T4>;
    EXPECT_EQ(std::string("bdif"), typeSetAsString<T5>());

    using T6 = T5::Append<T5>;
    EXPECT_EQ(std::string("bdifbdif"), typeSetAsString<T6>());

    using T7 = T5::Append<char, long>;
    EXPECT_EQ(std::string("bdifcl"), typeSetAsString<T7>());

    using T8 = T5::Append<char>::Append<long>;
    EXPECT_EQ(std::string("bdifcl"), typeSetAsString<T8>());

    using T9 = T8::Remove<TypeList<>>;
    EXPECT_EQ(std::string("bdifcl"), typeSetAsString<T9>());

    using T10 = T8::Remove<std::string>;
    EXPECT_EQ(std::string("bdifcl"), typeSetAsString<T10>());

    using T11 = T8::Remove<char>::Remove<int>;
    EXPECT_EQ(std::string("bdfl"), typeSetAsString<T11>());

    using T12 = T8::Remove<char, int>;
    EXPECT_EQ(std::string("bdfl"), typeSetAsString<T12>());

    using T13 = T8::Remove<TypeList<char, int>>;
    EXPECT_EQ(std::string("bdfl"), typeSetAsString<T13>());

    // Apply
    FloatGrid g1;
    GridBase& base = g1;
    bool applied = false;
    EXPECT_TRUE((!TypeList<>::apply([&](const auto&) {}, base)));
    EXPECT_TRUE((!TypeList<Int32Grid>::apply([&](const auto&) {}, base)));
    EXPECT_TRUE((!TypeList<DoubleGrid>::apply([&](const auto&) {}, base)));
    EXPECT_TRUE((!TypeList<DoubleGrid>::apply([&](auto&) {}, base)));
    EXPECT_TRUE((TypeList<FloatGrid>::apply([&](const auto& typed) {
        EXPECT_EQ(reinterpret_cast<const void*>(&g1), reinterpret_cast<const void*>(&typed));
        applied = true; }, base)));
    EXPECT_TRUE((applied));

    // check arg is passed non-const
    applied = false;
    EXPECT_TRUE((TypeList<FloatGrid>::apply([&](auto& typed) {
        EXPECT_EQ(reinterpret_cast<const void*>(&g1), reinterpret_cast<const void*>(&typed));
        applied = true; }, base)));
    EXPECT_TRUE(applied);

    applied = false;
    EXPECT_TRUE((TypeList<Int32Grid, FloatGrid, DoubleGrid>::apply([&](const auto& typed) {
            EXPECT_EQ(reinterpret_cast<const void*>(&g1), reinterpret_cast<const void*>(&typed));
            applied = true;
        }, base)));
    EXPECT_TRUE(applied);

    // check const args
    applied = false;
    const GridBase& cbase = base;
    EXPECT_TRUE((TypeList<Int32Grid, FloatGrid, DoubleGrid>::apply([&](const auto& typed) {
            EXPECT_EQ(reinterpret_cast<const void*>(&g1), reinterpret_cast<const void*>(&typed));
            applied = true;
        }, cbase)));
    EXPECT_TRUE(applied);

    // check same functor is used and how many types are processed
    struct Counter {
        Counter() = default;
        // delete copy constructor to check functor isn't being copied
        Counter(const Counter&) = delete;
        inline void operator()(const FloatGrid&) { ++mCounter; }
        int32_t mCounter = 0;
    } count;

    EXPECT_TRUE((TypeList<FloatGrid>::apply(std::ref(count), cbase)));
    EXPECT_EQ(count.mCounter, 1);
    EXPECT_TRUE((TypeList<FloatGrid, FloatGrid>::apply(std::ref(count), cbase)));
    EXPECT_EQ(count.mCounter, 2);
}


TEST_F(TestTypeList, testTupleList)
{
    using IntTypes = TupleList<Int16, Int32, Int64>;
    using EmptyList = TupleList<>;

    // To TypeList
    static_assert(std::is_same<IntTypes::AsTypeList, TypeList<Int16, Int32, Int64>>::value);
    static_assert(std::is_same<EmptyList::AsTypeList, TypeList<>>::value);

    IntTypes ints; // default init
    static_assert(ints.size() == 3);

    // Test foreach
    int counter = 1;
    ints.foreach([&](auto& i) {
        i = static_cast<typename std::decay<decltype(i)>::type>(counter++);
    });
    EXPECT_EQ(ints.get<0>(), 1);
    EXPECT_EQ(ints.get<1>(), 2);
    EXPECT_EQ(ints.get<2>(), 3);

    // Test evalFirstPred
    ints.evalFirstPred(
        [](auto Idx) { return Idx == 1; },
        [&](auto& i) { i = 7; });

    EXPECT_EQ(ints.get<0>(), 1);
    EXPECT_EQ(ints.get<1>(), 7);
    EXPECT_EQ(ints.get<2>(), 3);

    auto last = ints.evalFirstPred(
        [](auto Idx) { return Idx == 2; },
        [&](auto i) { return i; },
        Int64(-1));

    static_assert(std::is_same<decltype(last), Int64>::value);
    EXPECT_EQ(last, 3);

    // test arg init
    const IntTypes other(4,5,6);
    EXPECT_EQ(other.get<0>(), 4);
    EXPECT_EQ(other.get<1>(), 5);
    EXPECT_EQ(other.get<2>(), 6);

    // Test copying
    ints = other;
    EXPECT_EQ(ints.get<0>(), 4);
    EXPECT_EQ(ints.get<1>(), 5);
    EXPECT_EQ(ints.get<2>(), 6);

    // Test empty
    EmptyList empty;
    static_assert(empty.size() == 0);

    empty.foreach([](auto){});
    empty.evalFirstPred([](auto){}, [](auto){});
    auto ret = empty.evalFirstPred([](auto){}, [](auto){}, false);
    static_assert(std::is_same<decltype(ret), bool>::value);
    EXPECT_EQ(ret, false);
}


template <typename T> struct IsRegistered { inline void operator()() { EXPECT_TRUE(T::isRegistered()); } };
template <typename T> struct IsRegisteredType { inline void operator()() { EXPECT_TRUE(T::isRegisteredType()); } };
template <typename GridT> struct GridListContains { inline void operator()() { static_assert((GridTypes::Contains<GridT>)); } };
template <typename GridT> struct AttrListContains { inline void operator()() { static_assert((AttributeTypes::Contains<GridT>)); } };

TEST_F(TestTypeList, testOpenVDBTypeLists)
{
    openvdb::initialize();

#define CHECK_TYPE_LIST_IS_VALID(LIST_T) \
    static_assert((LIST_T::Size > 0));   \
    static_assert((std::is_same<LIST_T::Unique<>, LIST_T>::value));

    CHECK_TYPE_LIST_IS_VALID(GridTypes)
    CHECK_TYPE_LIST_IS_VALID(RealGridTypes)
    CHECK_TYPE_LIST_IS_VALID(IntegerGridTypes)
    CHECK_TYPE_LIST_IS_VALID(NumericGridTypes)
    CHECK_TYPE_LIST_IS_VALID(Vec3GridTypes)

    CHECK_TYPE_LIST_IS_VALID(TreeTypes)
    CHECK_TYPE_LIST_IS_VALID(RealTreeTypes)
    CHECK_TYPE_LIST_IS_VALID(IntegerTreeTypes)
    CHECK_TYPE_LIST_IS_VALID(NumericTreeTypes)
    CHECK_TYPE_LIST_IS_VALID(Vec3TreeTypes)

    static_assert((GridTypes::Size == TreeTypes::Size));
    static_assert((RealGridTypes::Size == RealTreeTypes::Size));
    static_assert((IntegerGridTypes::Size == IntegerTreeTypes::Size));
    static_assert((NumericGridTypes::Size == NumericTreeTypes::Size));
    static_assert((Vec3GridTypes::Size == Vec3TreeTypes::Size));

    GridTypes::foreach<IsRegistered>();

    RealGridTypes::foreach<GridListContains>();
    IntegerGridTypes::foreach<GridListContains>();
    NumericGridTypes::foreach<GridListContains>();
    Vec3GridTypes::foreach<GridListContains>();

    CHECK_TYPE_LIST_IS_VALID(AttributeTypes)
    CHECK_TYPE_LIST_IS_VALID(IntegerAttributeTypes)
    CHECK_TYPE_LIST_IS_VALID(NumericAttributeTypes)
    CHECK_TYPE_LIST_IS_VALID(Vec3AttributeTypes)
    CHECK_TYPE_LIST_IS_VALID(Mat3AttributeTypes)
    CHECK_TYPE_LIST_IS_VALID(Mat4AttributeTypes)
    CHECK_TYPE_LIST_IS_VALID(QuatAttributeTypes)

    AttributeTypes::foreach<IsRegistered>();

    RealAttributeTypes::foreach<AttrListContains>();
    IntegerAttributeTypes::foreach<AttrListContains>();
    NumericAttributeTypes::foreach<AttrListContains>();
    Vec3AttributeTypes::foreach<AttrListContains>();
    Mat3AttributeTypes::foreach<AttrListContains>();
    Mat4AttributeTypes::foreach<AttrListContains>();
    QuatAttributeTypes::foreach<AttrListContains>();

    CHECK_TYPE_LIST_IS_VALID(MapTypes)

    MapTypes::foreach<IsRegistered>();

    CHECK_TYPE_LIST_IS_VALID(MetaTypes)

    MetaTypes::foreach<IsRegisteredType>();

    // Test apply methods
    const FloatGrid grid;
    const GridBase& gridBase = grid;
    EXPECT_TRUE(GridTypes::apply([](const auto&) {}, gridBase));

    const FloatTree tree;
    const TreeBase& treeBase = tree;
    EXPECT_TRUE(TreeTypes::apply([](const auto&) {}, treeBase));

    openvdb::uninitialize();
}
