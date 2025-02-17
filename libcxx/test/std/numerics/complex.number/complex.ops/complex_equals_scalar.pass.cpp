//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   bool
//   operator==(const complex<T>& lhs, const T& rhs);

#include <complex>
#include <cassert>

#include "test_macros.h"

template <class T>
TEST_CONSTEXPR_CXX20
void
test_constexpr()
{
#if TEST_STD_VER > 11
    {
    constexpr std::complex<T> lhs(1.5, 2.5);
    constexpr T rhs(-2.5);
    static_assert(!(lhs == rhs), "");
    }
    {
    constexpr std::complex<T> lhs(1.5, 0);
    constexpr T rhs(-2.5);
    static_assert(!(lhs == rhs), "");
    }
    {
    constexpr std::complex<T> lhs(1.5, 2.5);
    constexpr T rhs(1.5);
    static_assert(!(lhs == rhs), "");
    }
    {
    constexpr std::complex<T> lhs(1.5, 0);
    constexpr T rhs(1.5);
    static_assert( (lhs == rhs), "");
    }
#endif
}

template <class T>
TEST_CONSTEXPR_CXX20
bool
test()
{
    {
    std::complex<T> lhs(1.5,  2.5);
    T rhs(-2.5);
    assert(!(lhs == rhs));
    }
    {
    std::complex<T> lhs(1.5, 0);
    T rhs(-2.5);
    assert(!(lhs == rhs));
    }
    {
    std::complex<T> lhs(1.5, 2.5);
    T rhs(1.5);
    assert(!(lhs == rhs));
    }
    {
    std::complex<T> lhs(1.5, 0);
    T rhs(1.5);
    assert( (lhs == rhs));
    }

    test_constexpr<T> ();
    return true;
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();

#if TEST_STD_VER > 17
    static_assert(test<float>());
    static_assert(test<double>());
    static_assert(test<long double>());
#endif

  return 0;
}
