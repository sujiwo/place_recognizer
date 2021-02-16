/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2014, Itseez Inc, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef KDTREE_H
#define KDTREE_H

#include "opencv2/core.hpp"

namespace cv
{
namespace ml
{

/*!
 Fast Nearest Neighbor Search Class.

 The class implements D. Lowe BBF (Best-Bin-First) algorithm for the last
 approximate (or accurate) nearest neighbor search in multi-dimensional spaces.

 First, a set of vectors is passed to KDTree::KDTree() constructor
 or KDTree::build() method, where it is reordered.

 Then arbitrary vectors can be passed to KDTree::findNearest() methods, which
 find the K nearest neighbors among the vectors from the initial set.
 The user can balance between the speed and accuracy of the search by varying Emax
 parameter, which is the number of leaves that the algorithm checks.
 The larger parameter values yield more accurate results at the expense of lower processing speed.

 \code
 KDTree T(points, false);
 const int K = 3, Emax = INT_MAX;
 int idx[K];
 float dist[K];
 T.findNearest(query_vec, K, Emax, idx, 0, dist);
 CV_Assert(dist[0] <= dist[1] && dist[1] <= dist[2]);
 \endcode
*/
class CV_EXPORTS_W KDTree
{
public:
    /*!
        The node of the search tree.
    */
    struct Node
    {
        Node() : idx(-1), left(-1), right(-1), boundary(0.f) {}
        Node(int _idx, int _left, int _right, float _boundary)
            : idx(_idx), left(_left), right(_right), boundary(_boundary) {}

        //! split dimension; >=0 for nodes (dim), < 0 for leaves (index of the point)
        int idx;
        //! node indices of the left and the right branches
        int left, right;
        //! go to the left if query_vec[node.idx]<=node.boundary, otherwise go to the right
        float boundary;
    };

    //! the default constructor
    CV_WRAP KDTree();
    //! the full constructor that builds the search tree
    CV_WRAP KDTree(InputArray points, bool copyAndReorderPoints = false);
    //! the full constructor that builds the search tree
    CV_WRAP KDTree(InputArray points, InputArray _labels,
                   bool copyAndReorderPoints = false);
    //! builds the search tree
    CV_WRAP void build(InputArray points, bool copyAndReorderPoints = false);
    //! builds the search tree
    CV_WRAP void build(InputArray points, InputArray labels,
                       bool copyAndReorderPoints = false);
    //! finds the K nearest neighbors of "vec" while looking at Emax (at most) leaves
    CV_WRAP int findNearest(InputArray vec, int K, int Emax,
                            OutputArray neighborsIdx,
                            OutputArray neighbors = noArray(),
                            OutputArray dist = noArray(),
                            OutputArray labels = noArray()) const;
    //! finds all the points from the initial set that belong to the specified box
    CV_WRAP void findOrthoRange(InputArray minBounds,
                                InputArray maxBounds,
                                OutputArray neighborsIdx,
                                OutputArray neighbors = noArray(),
                                OutputArray labels = noArray()) const;
    //! returns vectors with the specified indices
    CV_WRAP void getPoints(InputArray idx, OutputArray pts,
                           OutputArray labels = noArray()) const;
    //! return a vector with the specified index
    const float* getPoint(int ptidx, int* label = 0) const;
    //! returns the search space dimensionality
    CV_WRAP int dims() const;

    std::vector<Node> nodes; //!< all the tree nodes
    CV_PROP Mat points; //!< all the points. It can be a reordered copy of the input vector set or the original vector set.
    CV_PROP std::vector<int> labels; //!< the parallel array of labels.
    CV_PROP int maxDepth; //!< maximum depth of the search tree. Do not modify it
    CV_PROP_RW int normType; //!< type of the distance (cv::NORM_L1 or cv::NORM_L2) used for search. Initially set to cv::NORM_L2, but you can modify it
};

}
}

#endif
