/*
 * KMedians.cpp
 *
 *  Created on: Dec 2, 2020
 *      Author: sujiwo
 */

#include <exception>
#include <KMedians.h>

using namespace std;


namespace PlaceRecognizer {

/*
static int
kmedians(int nclusters, int nrows, int ncolumns, double** data, int** mask,
    double weight[], int transpose, int npass, char dist,
    double** cdata, int** cmask, int clusterid[], double* error,
    int tclusterid[], int counts[], int mapping[], double cache[])
*/

/*
Purpose
=======

The kcluster routine performs k-means or k-median clustering on a given set of
elements, using the specified distance measure. The number of clusters is given
by the user. Multiple passes are being made to find the optimal clustering
solution, each time starting from a different initial clustering.


Arguments
=========

nclusters  (input) int
The number of clusters to be found.

data       (input) double[nrows][ncolumns]
The array containing the data of the elements to be clustered (i.e., the gene
expression data).

mask       (input) int[nrows][ncolumns]
This array shows which data values are missing. If
mask[i][j] == 0, then data[i][j] is missing.

nrows      (input) int
The number of rows in the data matrix, equal to the number of genes.

ncolumns   (input) int
The number of columns in the data matrix, equal to the number of samples.

weight     (input) double[ncolumns] if transpose == 0,
                   double[nrows]    otherwise
The weights that are used to calculate the distance. This is equivalent
to including the jth data point weight[j] times in the calculation. The
weights can be non-integer.

transpose  (input) int
If transpose == 0, the rows of the matrix are clustered. Otherwise, columns
of the matrix are clustered.

npass      (input) int
The number of times clustering is performed. Clustering is performed npass
times, each time starting from a different (random) initial assignment of
genes to clusters. The clustering solution with the lowest within-cluster sum
of distances is chosen.
If npass == 0, then the clustering algorithm will be run once, where the
initial assignment of elements to clusters is taken from the clusterid array.

method     (input) char
Defines whether the arithmetic mean (method == 'a') or the median
(method == 'm') is used to calculate the cluster center.

dist       (input) char
Defines which distance measure is used, as given by the table:
dist == 'e': Euclidean distance
dist == 'b': City-block distance
dist == 'c': correlation
dist == 'a': absolute value of the correlation
dist == 'u': uncentered correlation
dist == 'x': absolute uncentered correlation
dist == 's': Spearman's rank correlation
dist == 'k': Kendall's tau
For other values of dist, the default (Euclidean distance) is used.

clusterid  (output; input) int[nrows] if transpose == 0
                           int[ncolumns] otherwise
The cluster number to which a gene or microarray was assigned. If npass == 0,
then on input clusterid contains the initial clustering assignment from which
the clustering algorithm starts. On output, it contains the clustering solution
that was found.

error      (output) double*
The sum of distances to the cluster center of each item in the optimal k-means
clustering solution that was found.

ifound     (output) int*
The number of times the optimal clustering solution was
found. The value of ifound is at least 1; its maximum value is npass. If the
number of clusters is larger than the number of elements being clustered,
*ifound is set to 0 as an error code. If a memory allocation error occurs,
*ifound is set to -1.

========================================================================
*/


KMedians::KMedians(int K, int iter) :
	numOfClusters(),
	iteration(iter)
{}


bool
KMedians::cluster(cv::Mat &binary_data)
{
	// Check & preparation
	// Make sure data type is unsigned 8-bit integer
	// with bit-width as multiples of 64
	int m_type = binary_data.type() & CV_MAT_DEPTH_MASK;
	assert(m_type==cv::DataType<uint8_t>::type);
	assert(binary_data.channels()==1);

	samples = binary_data;
	bit_width = samples.cols * 8;
	if (bit_width%64 != 0)
		throw runtime_error("Matrix column is not multiples of 64-bit");


	return true;
}

} /* namespace PlaceRecognizer */
