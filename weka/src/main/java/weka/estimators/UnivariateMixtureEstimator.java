/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    UnivariateMixtureEstimator.java
 *    Copyright (C) 2014 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.estimators;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.core.ContingencyTables;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Statistics;
import weka.core.Utils;

/**
 * Simple weighted mixture density estimator. Uses a mixture of Gaussians
 * and applies the leave-one-out bootstrap for model selection. Can alternatively use normalized entropy.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public class UnivariateMixtureEstimator implements UnivariateDensityEstimator, UnivariateIntervalEstimator, UnivariateQuantileEstimator, OptionHandler, Serializable {

	/** Constant for normal distribution. */
	private static double m_normConst = Math.log(Math.sqrt(2 * Math.PI));

	/**
	 * Fast univariate mixture model implementation.
	 */
	public class MM {

		/** Means */
		protected double[] m_Means = null;

		/** Standard deviations */
		protected double[] m_StdDevs = null;

		/** The priors, on log scale */
		protected double[] m_LogPriors = null;

		/** The number of actual components */
		protected int m_K;

		/**
		 * Returns string describing the estimator.
		 */
		@Override
		public String toString() {

			StringBuffer sb = new StringBuffer();
			sb.append("Mixture model estimator\n\n");
			for (int i = 0; i < this.m_LogPriors.length; i++) {
				sb.append("Mean: " + this.m_Means[i] + "\tStd. dev.: " + this.m_StdDevs[i] + "\tPrior prob.: " + Math.exp(this.m_LogPriors[i]) + "\n");
			}

			return sb.toString();
		}

		/**
		 * Returns smallest distance to given elements.
		 * Assumes m_Means has at least one element (i.e. m_K >= 1).
		 */
		protected double smallestDistance(final double val) {

			double min = Math.abs(val - this.m_Means[0]);
			for (int i = 1; i < this.m_K; i++) {
				if (Math.abs(val - this.m_Means[i]) < min) {
					min = Math.abs(val - this.m_Means[i]);
				}
			}
			return min;
		}

		/**
		 * Returns the index of the nearest mean.
		 * Assumes m_Means has at least one element (i.e. m_K >= 1).
		 */
		protected int nearestMean(final double val) {

			double min = Math.abs(val - this.m_Means[0]);
			int index = 0;
			for (int i = 1; i < this.m_K; i++) {
				if (Math.abs(val - this.m_Means[i]) < min) {
					min = Math.abs(val - this.m_Means[i]);
					index = i;
				}
			}
			return index;
		}

		/**
		 * Initializes the model. Assumes K >= 1, values.length >= 1,
		 * and values.length = weights.length.
		 */
		public void initializeModel(final int K, final double[] values, final double[] weights, final Random r) {

			// Initialize means using farthest points
			this.m_Means = new double[K];

			// Randomly choose first point
			double furthestVal = values[r.nextInt(values.length)];

			// Find K maximally distant points (if possible)
			this.m_K = 0;
			do {
				this.m_Means[this.m_K] = furthestVal;
				this.m_K++;
				if (this.m_K >= K) {
					break;
				}
				double maxMinDist = this.smallestDistance(values[0]);
				furthestVal = values[0];
				for (int i = 1; i < values.length; i++) {
					double minDist = this.smallestDistance(values[i]);
					if (minDist > maxMinDist) {
						maxMinDist = minDist;
						furthestVal = values[i];
					}
				}
				if (maxMinDist <= 0) {
					break;
				}
			} while (true);

			// Shrink array of means if necessary
			if (this.m_K < K) {
				double[] tempMeans = new double[this.m_K];
				System.arraycopy(this.m_Means, 0, tempMeans, 0, this.m_K);
				this.m_Means = tempMeans;
			}

			// Establish initial cluster assignments
			double[][] probs = new double[this.m_K][values.length];
			for (int i = 0; i < values.length; i++) {
				probs[this.nearestMean(values[i])][i] = 1.0;
			}

			// Compute initial parameters
			this.m_StdDevs = new double[this.m_K];
			this.m_LogPriors = new double[this.m_K];
			this.estimateParameters(values, weights, probs);
		}

		/**
		 * Estimate parameters.
		 */
		protected void estimateParameters(final double[] values, final double[] weights, final double[][] probs) {

			double totalSumOfWeights = 0;
			for (int j = 0; j < this.m_K; j++) {
				double sum = 0;
				double sumWeights = 0;
				for (int i = 0; i < values.length; i++) {
					double weight = probs[j][i] * weights[i];
					sum += weight * values[i];
					sumWeights += weight;
				}
				if (sumWeights <= 0) {
					this.m_Means[j] = 0;
				} else {
					this.m_Means[j] = sum / sumWeights;
				}
				totalSumOfWeights += sumWeights;
			}

			for (int j = 0; j < this.m_K; j++) {
				double sum = 0;
				double sumWeights = 0;
				for (int i = 0; i < values.length; i++) {
					double weight = probs[j][i] * weights[i];
					double diff = values[i] - this.m_Means[j];
					sum += weight * diff * diff;
					sumWeights += weight;
				}
				if ((sum <= 0) || (sumWeights <= 0)) {
					this.m_StdDevs[j] = 1.0e-6; // Hack to prevent unpleasantness
				} else {
					this.m_StdDevs[j] = Math.sqrt(sum / sumWeights);
					if (this.m_StdDevs[j] < 1.0e-6) {
						this.m_StdDevs[j] = 1.0e-6;
					}
				}
				if (sumWeights <= 0) {
					this.m_LogPriors[j] = -Double.MAX_VALUE;
				} else {
					this.m_LogPriors[j] = Math.log(sumWeights / totalSumOfWeights);
				}
			}
		}

		/**
		 * Computes loglikelihood of current model.
		 */
		public double loglikelihood(final double[] values, final double[] weights) {

			double sum = 0;
			double sumOfWeights = 0;
			for (int i = 0; i < values.length; i++) {
				sum += weights[i] * this.logDensity(values[i]);
				sumOfWeights += weights[i];
			}
			return sum / sumOfWeights;
		}

		/**
		 * Returns average of squared errors for current model.
		 */
		public double MSE() {

			double mse = 0;
			for (int i = 0; i < this.m_K; i++) {
				mse += this.m_StdDevs[i] * this.m_StdDevs[i] * Math.exp(this.m_LogPriors[i]);
			}
			return mse;
		}

		/**
		 * Density function of normal distribution.
		 */
		protected double logNormalDens(final double x, final double mean, final double stdDev) {

			double diff = x - mean;
			return -(diff * diff / (2 * stdDev * stdDev)) - m_normConst - Math.log(stdDev);
		}

		/**
		 * Joint densities per cluster.
		 */
		protected double[] logJointDensities(final double value) {

			double[] a = new double[this.m_K];
			for (int i = 0; i < this.m_K; i++) {
				a[i] = this.m_LogPriors[i] + this.logNormalDens(value, this.m_Means[i], this.m_StdDevs[i]);
			}
			return a;
		}

		/**
		 * Computes log of density for given value.
		 */
		public double logDensity(final double value) {

			double[] a = this.logJointDensities(value);
			double max = a[Utils.maxIndex(a)];
			double sum = 0.0;
			for (int i = 0; i < a.length; i++) {
				sum += Math.exp(a[i] - max);
			}

			return max + Math.log(sum);
		}

		/**
		 * Returns the interval for the given confidence value.
		 *
		 * @param conf the confidence value in the interval [0, 1]
		 * @return the interval
		 * @throws InterruptedException
		 */
		public double[][] predictIntervals(final double conf) throws InterruptedException {

			// Compute minimum and maximum value, and delta
			double val = Statistics.normalInverse(1.0 - (1.0 - conf) / 2);
			double min = Double.MAX_VALUE;
			double max = -Double.MAX_VALUE;
			for (int i = 0; i < this.m_Means.length; i++) {
				double l = this.m_Means[i] - val * this.m_StdDevs[i];
				if (l < min) {
					min = l;
				}
				double r = this.m_Means[i] + val * this.m_StdDevs[i];
				if (r > max) {
					max = r;
				}
			}
			double delta = (max - min) / UnivariateMixtureEstimator.this.m_NumIntervals;

			// Create array with estimated probabilities
			double[] probabilities = new double[UnivariateMixtureEstimator.this.m_NumIntervals];
			double leftVal = Math.exp(this.logDensity(min));
			for (int i = 0; i < UnivariateMixtureEstimator.this.m_NumIntervals; i++) {
				double rightVal = Math.exp(this.logDensity(min + (i + 1) * delta));
				probabilities[i] = 0.5 * (leftVal + rightVal) * delta;
				leftVal = rightVal;
			}

			// Sort array based on area of bin estimates
			int[] sortedIndices = Utils.sort(probabilities);

			// Mark the intervals to use
			double sum = 0;
			boolean[] toUse = new boolean[probabilities.length];
			int k = 0;
			while ((sum < conf) && (k < toUse.length)) {
				toUse[sortedIndices[toUse.length - (k + 1)]] = true;
				sum += probabilities[sortedIndices[toUse.length - (k + 1)]];
				k++;
			}

			// Don't need probabilities anymore
			probabilities = null;

			// Create final list of intervals
			ArrayList<double[]> intervals = new ArrayList<double[]>();

			// The current interval
			double[] interval = null;

			// Calculate actual intervals
			boolean haveStartedInterval = false;
			for (int i = 0; i < UnivariateMixtureEstimator.this.m_NumIntervals; i++) {

				// Should the current bin be used?
				if (toUse[i]) {

					// Do we need to create a new interval?
					if (haveStartedInterval == false) {
						haveStartedInterval = true;
						interval = new double[2];
						interval[0] = min + i * delta;
					}

					// Regardless, we should update the upper boundary
					interval[1] = min + (i + 1) * delta;
				} else {

					// We need to finalize and store the last interval
					// if necessary.
					if (haveStartedInterval) {
						haveStartedInterval = false;
						intervals.add(interval);
					}
				}
			}

			// Add last interval if there is one
			if (haveStartedInterval) {
				intervals.add(interval);
			}

			return intervals.toArray(new double[0][0]);
		}

		/**
		 * Returns the quantile for the given percentage.
		 *
		 * @param percentage the percentage
		 * @return the quantile
		 */
		public double predictQuantile(final double percentage) {

			// Compute minimum and maximum value, and delta
			double valRight = Statistics.normalInverse(percentage);
			double valLeft = Statistics.normalInverse(0.001);
			double min = Double.MAX_VALUE;
			double max = -Double.MAX_VALUE;
			for (int i = 0; i < this.m_Means.length; i++) {
				double l = this.m_Means[i] - valLeft * this.m_StdDevs[i];
				if (l < min) {
					min = l;
				}
				double r = this.m_Means[i] + valRight * this.m_StdDevs[i];
				if (r > max) {
					max = r;
				}
			}
			double delta = (max - min) / UnivariateMixtureEstimator.this.m_NumIntervals;

			double sum = 0;
			double leftVal = Math.exp(this.logDensity(min));
			for (int i = 0; i < UnivariateMixtureEstimator.this.m_NumIntervals; i++) {
				if (sum >= percentage) {
					return min + i * delta;
				}
				double rightVal = Math.exp(this.logDensity(min + (i + 1) * delta));
				sum += 0.5 * (leftVal + rightVal) * delta;
				leftVal = rightVal;
			}
			return max;
		}
	}

	/** For serialization */
	private static final long serialVersionUID = -2035274930137353656L;

	/** The values used for this estimator */
	protected double[] m_Values = new double[1000];

	/** The weights used for this estimator */
	protected double[] m_Weights = new double[1000];

	/** The number of values that have been seen */
	protected int m_NumValues;

	/** The current mixture model */
	protected MM m_MixtureModel;

	/** The number of components to use (default is -1)*/
	protected int m_NumComponents = -1;

	/** The maximum number of components to use (default is 5) */
	protected int m_MaxNumComponents = 5;

	/** The random number seed to use (default is 1*/
	protected int m_Seed = 1;

	/** The number of Bootstrap runs to use to select the number of components (default is 10) */
	protected int m_NumBootstrapRuns = 10;

	/** The number of intervals used to approximate prediction interval. */
	protected int m_NumIntervals = 1000;

	/** Whether to use normalized entropy instance of bootstrap. */
	protected boolean m_UseNormalizedEntropy = false;

	/** Whether to output debug info. */
	protected boolean m_Debug = false;

	/** The random number generator. */
	protected Random m_Random = new Random(this.m_Seed);

	/**
	 * Returns a string describing the estimator.
	 */
	public String globalInfo() {
		return "Estimates a univariate mixture model.";
	}

	/**
	 * @return whether normalized entropy is used
	 */
	public boolean getUseNormalizedEntropy() {
		return this.m_UseNormalizedEntropy;
	}

	/**
	 * @param useNormalizedEntropy whether to use normalized entropy
	 */
	public void setUseNormalizedEntropy(final boolean useNormalizedEntropy) {
		this.m_UseNormalizedEntropy = useNormalizedEntropy;
	}

	/**
	 * The tool tip for this property.
	 */
	public String numBootstrapRunsToolTipText() {
		return "The number of Bootstrap runs to choose the number of components.";
	}

	/**
	 * Returns the number of Bootstrap runs.
	 *
	 * @return the number of Bootstrap runs
	 */
	public int getNumBootstrapRuns() {
		return this.m_NumBootstrapRuns;
	}

	/**
	 * Sets the number of Bootstrap runs.
	 *
	 * @param mnumBootstrapRuns the number of Bootstrap runs
	 */
	public void setNumBootstrapRuns(final int numBootstrapRuns) {
		this.m_NumBootstrapRuns = numBootstrapRuns;
	}

	/**
	 * The tool tip for this property.
	 */
	public String numComponentsToolTipText() {
		return "The number of mixture components to use.";
	}

	/**
	 * Returns the number of components to use.
	 *
	 * @return the m_NumComponents
	 */
	public int getNumComponents() {
		return this.m_NumComponents;
	}

	/**
	 * Sets the number of components to use.
	 *
	 * @param m_NumComponents the m_NumComponents to set
	 */
	public void setNumComponents(final int numComponents) {
		this.m_NumComponents = numComponents;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String seedTipText() {
		return "The random number seed to be used.";
	}

	/**
	 * Set the seed for random number generation.
	 *
	 * @param seed the seed
	 */
	public void setSeed(final int seed) {

		this.m_Seed = seed;
		this.m_Random = new Random(seed);
	}

	/**
	 * Gets the seed for the random number generations
	 *
	 * @return the seed for the random number generation
	 */
	public int getSeed() {

		return this.m_Seed;
	}

	/**
	 * The tool tip for this property.
	 */
	public String maxNumComponentsToolTipText() {
		return "The maximum number of mixture components to use.";
	}

	/**
	 * Returns the number of components to use.
	 *
	 * @return the maximum number of components to use
	 */
	public int getMaxNumComponents() {
		return this.m_MaxNumComponents;
	}

	/**
	 * Sets the number of components to use.
	 *
	 * @param maxNumComponents the maximum number of components to evaluate
	 */
	public void setMaxNumComponents(final int maxNumComponents) {
		this.m_MaxNumComponents = maxNumComponents;
	}

	/**
	 * Adds a value to the density estimator.
	 *
	 * @param value the value to add
	 * @param weight the weight of the value
	 */
	@Override
	public void addValue(final double value, final double weight) {

		// Do we need to add value at all?
		if (!Utils.eq(weight, 0)) {

			// Invalidate current model
			this.m_MixtureModel = null;

			// Do we need to expand the arrays?
			if (this.m_NumValues == this.m_Values.length) {
				double[] newWeights = new double[2 * this.m_NumValues];
				double[] newValues = new double[2 * this.m_NumValues];
				System.arraycopy(this.m_Values, 0, newValues, 0, this.m_NumValues);
				System.arraycopy(this.m_Weights, 0, newWeights, 0, this.m_NumValues);
				this.m_Values = newValues;
				this.m_Weights = newWeights;
			}

			// Add values
			this.m_Values[this.m_NumValues] = value;
			this.m_Weights[this.m_NumValues] = weight;
			this.m_NumValues++;
		}
	}

	/**
	 * Build mixture model. Assumes K >= 1, values.length >= 1,
	 * and values.length = weights.length.
	 */
	public MM buildModel(final int K, final double[] values, final double[] weights) {

		// Initialize model using k-means
		MM model = null;
		double bestMSE = Double.MAX_VALUE;
		int numAttempts = 0;
		while (numAttempts < 5) {

			// Initialize model
			MM tempModel = new UnivariateMixtureEstimator().new MM();
			tempModel.initializeModel(K, values, weights, this.m_Random);

			// Run k-means until MSE converges
			double oldMSE = Double.MAX_VALUE;
			double MSE = tempModel.MSE();

			if (this.m_Debug) {
				System.err.println("MSE: " + MSE);
			}

			double[][] probs = new double[tempModel.m_K][values.length];
			while (Utils.sm(MSE, oldMSE)) {

				// Compute memberships
				for (int j = 0; j < probs.length; j++) {
					Arrays.fill(probs[j], 0);
				}
				for (int i = 0; i < values.length; i++) {
					probs[tempModel.nearestMean(values[i])][i] = 1.0;
				}

				// Estimate parameters
				tempModel.estimateParameters(values, weights, probs);

				// Compute MSE for updated model
				oldMSE = MSE;
				MSE = tempModel.MSE();

				if (this.m_Debug) {
					System.err.println("MSE: " + MSE);
				}
			}
			if (MSE < bestMSE) {
				bestMSE = MSE;
				model = tempModel;
			}

			if (this.m_Debug) {
				System.err.println("Best MSE: " + bestMSE);
			}

			numAttempts++;
		}

		// Run until likelihood converges
		double oldLogLikelihood = -Double.MAX_VALUE;
		double loglikelihood = model.loglikelihood(values, weights);
		double[][] probs = new double[model.m_K][values.length];
		while (Utils.gr(loglikelihood, oldLogLikelihood)) {

			// Establish membership probabilities
			for (int i = 0; i < values.length; i++) {
				double[] p = Utils.logs2probs(model.logJointDensities(values[i]));
				for (int j = 0; j < p.length; j++) {
					probs[j][i] = p[j];
				}
			}

			// Estimate parameters
			model.estimateParameters(values, weights, probs);

			// Compute loglikelihood for updated model
			oldLogLikelihood = loglikelihood;
			loglikelihood = model.loglikelihood(values, weights);
		}

		return model;
	}

	/**
	 * Creates a new dataset of the same size using random sampling with
	 * replacement according to the given weight vector. The weights of the
	 * instances in the new dataset are set to one.
	 */
	public double[][] resampleWithWeights(final Random random, final boolean[] sampled) {

		// Walker's method, see pp. 232 of "Stochastic Simulation" by B.D. Ripley
		double[] P = new double[this.m_Weights.length];
		System.arraycopy(this.m_Weights, 0, P, 0, this.m_Weights.length);
		Utils.normalize(P);
		double[] Q = new double[this.m_Weights.length];
		int[] A = new int[this.m_Weights.length];
		int[] W = new int[this.m_Weights.length];
		int M = this.m_Weights.length;
		int NN = -1;
		int NP = M;
		for (int I = 0; I < M; I++) {
			if (P[I] < 0) {
				throw new IllegalArgumentException("Weights have to be positive.");
			}
			Q[I] = M * P[I];
			if (Q[I] < 1.0) {
				W[++NN] = I;
			} else {
				W[--NP] = I;
			}
		}
		if (NN > -1 && NP < M) {
			for (int S = 0; S < M - 1; S++) {
				int I = W[S];
				int J = W[NP];
				A[I] = J;
				Q[J] += Q[I] - 1.0;
				if (Q[J] < 1.0) {
					NP++;
				}
				if (NP >= M) {
					break;
				}
			}
			// A[W[M]] = W[M];
		}

		for (int I = 0; I < M; I++) {
			Q[I] += I;
		}

		// Do we need to keep track of how many copies to use?
		int[] counts = new int[M];

		int count = 0;
		for (int i = 0; i < this.m_Weights.length; i++) {
			int ALRV;
			double U = M * random.nextDouble();
			int I = (int) U;
			if (U < Q[I]) {
				ALRV = I;
			} else {
				ALRV = A[I];
			}
			counts[ALRV]++;
			if (!sampled[ALRV]) {
				sampled[ALRV] = true;
				count++;
			}
		}

		// Generate output
		double[][] output = new double[2][count];
		int index = 0;
		for (int i = 0; i < M; i++) {
			if (counts[i] > 0) {
				output[0][index] = this.m_Values[i];
				output[1][index] = counts[i];
				index++;
			}
		}

		return output;
	}

	/**
	 * Selects the number of components using leave-one-out Bootstrap, estimating loglikelihood.
	 *
	 * @return the number of components to use
	 */
	protected int findNumComponentsUsingBootStrap() {

		if (this.m_NumComponents > 0) {
			return this.m_NumComponents;
		}
		if (this.m_MaxNumComponents <= 1) {
			return 1;
		}

		double bestLogLikelihood = -Double.MAX_VALUE;
		int bestNumComponents = 1;
		for (int i = 1; i <= this.m_MaxNumComponents; i++) {
			double logLikelihood = 0;
			for (int k = 0; k < this.m_NumBootstrapRuns; k++) {
				boolean[] inBag = new boolean[this.m_NumValues];
				double[][] output = this.resampleWithWeights(this.m_Random, inBag);
				MM mixtureModel = this.buildModel(i, output[0], output[1]);
				double locLogLikelihood = 0;
				double totalWeight = 0;
				for (int j = 0; j < this.m_NumValues; j++) {
					if (!inBag[j]) {
						double weight = this.m_Weights[j];
						locLogLikelihood += weight * mixtureModel.logDensity(this.m_Values[j]);
						totalWeight += weight;
					}
				}
				locLogLikelihood /= totalWeight;
				logLikelihood += locLogLikelihood;
			}
			logLikelihood /= this.m_NumBootstrapRuns;
			if (this.m_Debug) {
				System.err.println("Loglikelihood: " + logLikelihood + "\tNumber of components: " + i);
			}
			if (logLikelihood > bestLogLikelihood) {
				bestNumComponents = i;
				bestLogLikelihood = logLikelihood;
			}
		}

		return bestNumComponents;
	}

	/**
	 * Calculates entrpy for given model and data.
	 */
	protected double entropy(final MM mixtureModel) {

		double entropy = 0;
		for (int j = 0; j < this.m_NumValues; j++) {
			entropy += this.m_Weights[j] * ContingencyTables.entropy(Utils.logs2probs(mixtureModel.logJointDensities(this.m_Values[j])));
		}
		entropy *= Utils.log2; // Need natural logarithm, not base-2 logarithm

		return entropy / this.m_NumValues;
	}

	/**
	 * Selects the number of components using normalized entropy.
	 *
	 * @return the model to use
	 */
	protected MM findModelUsingNormalizedEntropy() {

		if (this.m_NumComponents > 0) {
			return this.buildModel(this.m_NumComponents, this.m_Values, this.m_Weights);
		}
		if (this.m_MaxNumComponents <= 1) {
			return this.buildModel(1, this.m_Values, this.m_Weights);
		}

		// Loglikelihood for one cluster
		MM bestMixtureModel = this.buildModel(1, this.m_Values, this.m_Weights);
		double loglikelihoodForOneCluster = bestMixtureModel.loglikelihood(this.m_Values, this.m_Weights);
		double bestNormalizedEntropy = 1;
		for (int i = 2; i <= this.m_MaxNumComponents; i++) {
			MM mixtureModel = this.buildModel(i, this.m_Values, this.m_Weights);

			double loglikelihood = mixtureModel.loglikelihood(this.m_Values, this.m_Weights);
			if (loglikelihood < loglikelihoodForOneCluster) {
				// This appears to happen in practice, hopefully not because of a bug...
				if (this.m_Debug) {
					System.err.println("Likelihood for one cluster greater than for " + i + " clusters.");
				}
				continue;
			}
			double entropy = this.entropy(mixtureModel);
			double normalizedEntropy = entropy / (loglikelihood - loglikelihoodForOneCluster);

			if (this.m_Debug) {
				System.err.println(
						"Entropy: " + entropy + "\tLogLikelihood: " + loglikelihood + "\tLoglikelihood for one cluster: " + loglikelihoodForOneCluster + "\tNormalized entropy: " + normalizedEntropy + "\tNumber of components: " + i);
			}
			if (normalizedEntropy < bestNormalizedEntropy) {
				bestMixtureModel = mixtureModel;
				bestNormalizedEntropy = normalizedEntropy;
			}
		}

		return bestMixtureModel;
	}

	/**
	 * Updates the model based on the current data.
	 * Uses the leave-one-out Bootstrap to choose the number of components.
	 */
	protected void updateModel() {

		if (this.m_MixtureModel != null) {
			return;
		} else if (this.m_NumValues > 0) {

			// Shrink arrays if necessary
			if (this.m_Values.length > this.m_NumValues) {
				double[] values = new double[this.m_NumValues];
				double[] weights = new double[this.m_NumValues];
				System.arraycopy(this.m_Values, 0, values, 0, this.m_NumValues);
				System.arraycopy(this.m_Weights, 0, weights, 0, this.m_NumValues);
				this.m_Values = values;
				this.m_Weights = weights;
			}

			if (this.m_UseNormalizedEntropy) {
				this.m_MixtureModel = this.findModelUsingNormalizedEntropy();
			} else {
				this.m_MixtureModel = this.buildModel(this.findNumComponentsUsingBootStrap(), this.m_Values, this.m_Weights);
			}
		}
	}

	/**
	 * Returns the interval for the given confidence value.
	 *
	 * @param conf the confidence value in the interval [0, 1]
	 * @return the interval
	 * @throws InterruptedException
	 */
	@Override
	public double[][] predictIntervals(final double conf) throws InterruptedException {

		this.updateModel();

		return this.m_MixtureModel.predictIntervals(conf);
	}

	/**
	 * Returns the quantile for the given percentage.
	 *
	 * @param percentage the percentage
	 * @return the quantile
	 */
	@Override
	public double predictQuantile(final double percentage) {

		this.updateModel();

		return this.m_MixtureModel.predictQuantile(percentage);
	}

	/**
	 * Returns the natural logarithm of the density estimate at the given
	 * point.
	 *
	 * @param value the value at which to evaluate
	 * @return the natural logarithm of the density estimate at the given
	 * value
	 */
	@Override
	public double logDensity(final double value) {

		this.updateModel();
		if (this.m_MixtureModel == null) {
			return Math.log(Double.MIN_VALUE);
		}
		return this.m_MixtureModel.logDensity(value);
	}

	/**
	 * Returns textual description of this estimator.
	 */
	@Override
	public String toString() {

		this.updateModel();
		if (this.m_MixtureModel == null) {
			return "";
		}
		return this.m_MixtureModel.toString();
	}

	/**
	 * Returns an enumeration that lists the command-line options that are available
	 *
	 * @return the list of options as an enumeration
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> options = new Vector<Option>();
		options.addElement(new Option("\tNumber of components to use (default: -1).", "N", 1, "-N"));
		options.addElement(new Option("\tMaximum number of components to use (default: 5).", "M", 1, "-M"));
		options.addElement(new Option("\tSeed for the random number generator (default: 1).", "S", 1, "-S"));
		options.addElement(new Option("\tThe number of bootstrap runs to use (default: 10).", "B", 1, "-B"));
		options.addElement(new Option("\tUse normalized entropy instead of bootstrap.", "E", 1, "-E"));
		return options.elements();
	}

	/**
	 * Sets options based on the given array of strings.
	 *
	 * @param options the list of options to parse
	 */
	@Override
	public void setOptions(final String[] options) throws Exception {

		String optionString = Utils.getOption("N", options);
		if (optionString.length() > 0) {
			this.setNumComponents(Integer.parseInt(optionString));
		} else {
			this.setNumComponents(-1);
		}
		optionString = Utils.getOption("M", options);
		if (optionString.length() > 0) {
			this.setMaxNumComponents(Integer.parseInt(optionString));
		} else {
			this.setMaxNumComponents(5);
		}
		optionString = Utils.getOption("S", options);
		if (optionString.length() > 0) {
			this.setSeed(Integer.parseInt(optionString));
		} else {
			this.setSeed(1);
		}
		optionString = Utils.getOption("B", options);
		if (optionString.length() > 0) {
			this.setNumBootstrapRuns(Integer.parseInt(optionString));
		} else {
			this.setNumBootstrapRuns(10);
		}
		this.m_UseNormalizedEntropy = Utils.getFlag("E", options);
		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Returns the current set of options.
	 *
	 * @return the current set of options as a string
	 */
	@Override
	public String[] getOptions() {
		Vector<String> options = new Vector<String>();

		options.add("-N");
		options.add("" + this.getNumComponents());

		options.add("-M");
		options.add("" + this.getMaxNumComponents());

		options.add("-S");
		options.add("" + this.getSeed());

		options.add("-B");
		options.add("" + this.getNumBootstrapRuns());

		if (this.m_UseNormalizedEntropy) {
			options.add("-E");
		}

		return options.toArray(new String[0]);
	}

	/**
	 * Returns the revision string.
	 *
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 10971 $");
	}

	/**
	 * Main method, used for testing this class.
	 */
	public static void main(final String[] args) throws Exception {

		// Get random number generator initialized by system
		Random r = new Random();

		// Create density estimator
		UnivariateMixtureEstimator e = new UnivariateMixtureEstimator();
		e.setOptions(Arrays.copyOf(args, args.length));

		// Output the density estimator
		System.out.println(e);

		// Monte Carlo integration
		double sum = 0;
		for (int i = 0; i < 100000; i++) {
			sum += Math.exp(e.logDensity(r.nextDouble() * 10.0 - 5.0));
		}
		System.out.println("Approximate integral: " + 10.0 * sum / 100000);

		// Add Gaussian values into it
		for (int i = 0; i < 100000; i++) {
			e.addValue(r.nextGaussian() * 0.5 - 1, 1);
			e.addValue(r.nextGaussian() * 0.5 + 1, 3);
		}

		// Output the density estimator
		System.out.println(e);

		// Monte Carlo integration
		sum = 0;
		for (int i = 0; i < 100000; i++) {
			sum += Math.exp(e.logDensity(r.nextDouble() * 10.0 - 5.0));
		}
		System.out.println("Approximate integral: " + 10.0 * sum / 100000);

		// Create density estimator
		e = new UnivariateMixtureEstimator();
		e.setOptions(Arrays.copyOf(args, args.length));

		// Add Gaussian values into it
		for (int i = 0; i < 100000; i++) {
			e.addValue(r.nextGaussian() * 0.5 - 1, 1);
			e.addValue(r.nextGaussian() * 0.5 + 1, 1);
			e.addValue(r.nextGaussian() * 0.5 + 1, 1);
			e.addValue(r.nextGaussian() * 0.5 + 1, 1);
		}

		// Output the density estimator
		System.out.println(e);

		// Monte Carlo integration
		sum = 0;
		for (int i = 0; i < 100000; i++) {
			sum += Math.exp(e.logDensity(r.nextDouble() * 10.0 - 5.0));
		}
		System.out.println("Approximate integral: " + 10.0 * sum / 100000);

		// Create density estimator
		e = new UnivariateMixtureEstimator();
		e.setOptions(Arrays.copyOf(args, args.length));

		// Add Gaussian values into it
		for (int i = 0; i < 100000; i++) {
			e.addValue(r.nextGaussian() * 5.0 + 3.0, 1);
		}

		// Output the density estimator
		System.out.println(e);

		// Check interval estimates
		double[][] intervals = e.predictIntervals(0.95);
		System.out.println("Lower: " + intervals[0][0] + " Upper: " + intervals[0][1]);
		double covered = 0;
		for (int i = 0; i < 100000; i++) {
			double val = r.nextGaussian() * 5.0 + 3.0;
			if (val >= intervals[0][0] && val <= intervals[0][1]) {
				covered++;
			}
		}
		System.out.println("Coverage: " + covered / 100000);

		intervals = e.predictIntervals(0.8);
		System.out.println("Lower: " + intervals[0][0] + " Upper: " + intervals[0][1]);
		covered = 0;
		for (int i = 0; i < 100000; i++) {
			double val = r.nextGaussian() * 5.0 + 3.0;
			if (val >= intervals[0][0] && val <= intervals[0][1]) {
				covered++;
			}
		}
		System.out.println("Coverage: " + covered / 100000);

		// Output quantile
		System.out.println("95% quantile: " + e.predictQuantile(0.95));
	}
}