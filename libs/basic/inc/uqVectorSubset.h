/* uq/libs/queso/inc/uqVectorSubset.h
 *
 * Copyright (C) 2008 The QUESO Team, http://queso.ices.utexas.edu/
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#ifndef __UQ_VECTOR_SUBSET_H__
#define __UQ_VECTOR_SUBSET_H__

#include <uqVectorSpace.h>

//*****************************************************
// Base class
//*****************************************************
template <class V, class M>
class uqVectorSubsetClass : public uqVectorSetClass<V,M>
{
public:
           uqVectorSubsetClass();
           uqVectorSubsetClass(const uqBaseEnvironmentClass&  env,
                               const char*                    prefix,
                               const uqVectorSpaceClass<V,M>& vectorSpace);
  virtual ~uqVectorSubsetClass();

           const uqVectorSpaceClass<V,M>& vectorSpace()                 const;
  virtual        bool                     contains   (const V& vec)     const = 0;
  virtual        void                     print      (std::ostream& os) const;

protected:
  using uqVectorSetClass<V,M>::m_env;
  using uqVectorSetClass<V,M>::m_prefix;

  const uqVectorSpaceClass<V,M>* m_vectorSpace;
};

template <class V, class M>
uqVectorSubsetClass<V,M>::uqVectorSubsetClass()
  :
  uqVectorSetClass<V,M>(),
  m_vectorSpace        (NULL)
{
  UQ_FATAL_TEST_MACRO(true,
                      m_env.rank(),
                      "uqVectorSubsetClass<V,M>::constructor(), default",
                      "should not be used by user");
}

template <class V, class M>
uqVectorSubsetClass<V,M>::uqVectorSubsetClass(
  const uqBaseEnvironmentClass&  env,
  const char*                    prefix,
  const uqVectorSpaceClass<V,M>& vectorSpace)
  :
  uqVectorSetClass<V,M>(env,prefix,0.),
  m_vectorSpace        (&vectorSpace)
{
  if ((m_env.verbosity() >= 5) && (m_env.rank() == 0)) {
    std::cout << "Entering uqVectorSubsetClass<V,M>::constructor()"
              << std::endl;
  }

  if ((m_env.verbosity() >= 5) && (m_env.rank() == 0)) {
    std::cout << "Leaving uqVectorSubsetClass<V,M>::constructor()"
              << std::endl;
  }
}

template <class V, class M>
uqVectorSubsetClass<V,M>::~uqVectorSubsetClass()
{
  //std::cout << "Entering uqVectorSubsetClass<V,M>::destructor()"
  //          << std::endl;

  //std::cout << "Leaving uqVectorSubsetClass<V,M>::destructor()"
  //          << std::endl;
}

template <class V, class M>
const uqVectorSpaceClass<V,M>&
uqVectorSubsetClass<V,M>::vectorSpace() const
{
  return *m_vectorSpace;
}

template <class V, class M>
void
uqVectorSubsetClass<V,M>::print(std::ostream& os) const
{
  return;
}

//*****************************************************
// Box class
//*****************************************************
template<class V, class M>
class uqBoxSubsetClass : public uqVectorSubsetClass<V,M> {
public:
  uqBoxSubsetClass(const uqBaseEnvironmentClass&  env,
                   const char*                    prefix,
                   const uqVectorSpaceClass<V,M>& vectorSpace,
                   const V&                       minValues,
                   const V&                       maxValues);
 ~uqBoxSubsetClass();

  bool contains(const V& vec)     const;
  void print   (std::ostream& os) const;

protected:
  using uqVectorSetClass   <V,M>::m_env;
  using uqVectorSetClass   <V,M>::m_prefix;
  using uqVectorSetClass   <V,M>::m_volume;
  using uqVectorSubsetClass<V,M>::m_vectorSpace;

  V m_minValues;
  V m_maxValues;
};

template<class V, class M>
uqBoxSubsetClass<V,M>::uqBoxSubsetClass(
  const uqBaseEnvironmentClass&  env,
  const char*                    prefix,
  const uqVectorSpaceClass<V,M>& vectorSpace,
  const V&                       minValues,
  const V&                       maxValues)
  :
  uqVectorSubsetClass<V,M>(env,prefix,vectorSpace),
  m_minValues(minValues),
  m_maxValues(maxValues)
{
  m_volume = 1.;
  for (unsigned int i = 0; i < m_vectorSpace->dim(); ++i) {
    m_volume *= (m_maxValues[i] - m_minValues[i]);
  }
}

template<class V, class M>
uqBoxSubsetClass<V,M>::~uqBoxSubsetClass()
{
}

template<class V, class M>
bool
uqBoxSubsetClass<V,M>::contains(const V& vec) const
{
  bool result = true;

  for (unsigned int i = 0; (i < m_vectorSpace->dim()) && (result == true); ++i) {
    result = (m_maxValues[i] <= vec[i]) && (vec[i] <= m_minValues[i]);
  }

  return result;
}

template <class V, class M>
void
uqBoxSubsetClass<V,M>::print(std::ostream& os) const
{
  return;
}
#endif // __UQ_VECTOR_SUBSET_H__
