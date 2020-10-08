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
 *    LogisticBase.java
 *    Copyright (C) 2003-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.lmt;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

/**
 * Base/helper class for building logistic regression models with the LogitBoost algorithm. Used for building logistic model trees (weka.classifiers.trees.lmt.LMT) and standalone logistic regression
 * (weka.classifiers.functions.SimpleLogistic).
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Niels Landwehr
 * @author Marc Sumner
 * @version $Revision$
 */
public class LogisticBase extends AbstractClassifier implements WeightedInstancesHandler {

	/** for serialization */
	static final long serialVersionUID = 168765678097825064L;

	/** Header-only version of the numeric version of the training data */
	protected Instances m_numericDataHeader;
	/**
	 * Numeric version of the training data. Original class is replaced by a numeric pseudo-class.
	 */
	protected Instances m_numericData;

	/** Training data */
	protected Instances m_train;

	/** Use cross-validation to determine best number of LogitBoost iterations ? */
	protected boolean m_useCrossValidation;

	/** Use error on probabilities for stopping criterion of LogitBoost? */
	protected boolean m_errorOnProbabilities;

	/**
	 * Use fixed number of iterations for LogitBoost? (if negative, cross-validate number of iterations)
	 */
	protected int m_fixedNumIterations;

	/**
	 * Use heuristic to stop performing LogitBoost iterations earlier? If enabled, LogitBoost is stopped if the current (local) minimum of the error on a test set as a function of the number of iterations has not changed for m_heuristicStop
	 * iterations.
	 */
	protected int m_heuristicStop = 50;

	/** The number of LogitBoost iterations performed. */
	protected int m_numRegressions = 0;

	/** The maximum number of LogitBoost iterations */
	protected int m_maxIterations;

	/** The number of different classes */
	protected int m_numClasses;

	/** Array holding the simple regression functions fit by LogitBoost */
	protected SimpleLinearRegression[][] m_regressions;

	/** Number of folds for cross-validating number of LogitBoost iterations */
	protected static int m_numFoldsBoosting = 5;

	/** Threshold on the Z-value for LogitBoost */
	protected static final double Z_MAX = 3;

	/** If true, the AIC is used to choose the best iteration */
	private boolean m_useAIC = false;

	/** Effective number of parameters used for AIC / BIC automatic stopping */
	protected double m_numParameters = 0;

	/**
	 * Threshold for trimming weights. Instances with a weight lower than this (as a percentage of total weights) are not included in the regression fit.
	 **/
	protected double m_weightTrimBeta = 0;

	/**
	 * Constructor that creates LogisticBase object with standard options.
	 */
	public LogisticBase() {
		this.m_fixedNumIterations = -1;
		this.m_useCrossValidation = true;
		this.m_errorOnProbabilities = false;
		this.m_maxIterations = 500;
		this.m_useAIC = false;
		this.m_numParameters = 0;
		this.m_numDecimalPlaces = 2;
	}

	/**
	 * Constructor to create LogisticBase object.
	 *
	 * @param numBoostingIterations
	 *            fixed number of iterations for LogitBoost (if negative, use cross-validation or stopping criterion on the training data).
	 * @param useCrossValidation
	 *            cross-validate number of LogitBoost iterations (if false, use stopping criterion on the training data).
	 * @param errorOnProbabilities
	 *            if true, use error on probabilities instead of misclassification for stopping criterion of LogitBoost
	 */
	public LogisticBase(final int numBoostingIterations, final boolean useCrossValidation, final boolean errorOnProbabilities) {
		this.m_fixedNumIterations = numBoostingIterations;
		this.m_useCrossValidation = useCrossValidation;
		this.m_errorOnProbabilities = errorOnProbabilities;
		this.m_maxIterations = 500;
		this.m_useAIC = false;
		this.m_numParameters = 0;
		this.m_numDecimalPlaces = 2;
	}

	/**
	 * Builds the logistic regression model usiing LogitBoost.
	 *
	 * @param data
	 *            the training data
	 * @throws Exception
	 *             if something goes wrong
	 */
	@Override
	public void buildClassifier(final Instances data) throws Exception {

		this.m_train = new Instances(data);

		this.m_numClasses = this.m_train.numClasses();

		// get numeric version of the training data (class variable replaced by
		// numeric pseudo-class)
		this.m_numericData = this.getNumericData(this.m_train);

		// init the array of simple regression functions
		this.m_regressions = this.initRegressions();
		this.m_numRegressions = 0;

		if (this.m_fixedNumIterations > 0) {
			// run LogitBoost for fixed number of iterations
			this.performBoosting(this.m_fixedNumIterations);
		} else if (this.m_useAIC) { // Marc had this after the test for
			// m_useCrossValidation. Changed by Eibe.
			// run LogitBoost using information criterion for stopping
			this.performBoostingInfCriterion();
		} else if (this.m_useCrossValidation) {
			// cross-validate number of LogitBoost iterations
			this.performBoostingCV();
		} else {
			// run LogitBoost with number of iterations that minimizes error on the
			// training set
			this.performBoosting();
		}

		// clean up
		this.cleanup();
	}

	/**
	 * Runs LogitBoost, determining the best number of iterations by cross-validation.
	 *
	 * @throws Exception
	 *             if something goes wrong
	 */
	protected void performBoostingCV() throws Exception {

		// completed iteration keeps track of the number of iterations that have
		// been
		// performed in every fold (some might stop earlier than others).
		// Best iteration is selected only from these.
		int completedIterations = this.m_maxIterations;

		Instances allData = new Instances(this.m_train);

		allData.stratify(m_numFoldsBoosting);

		double[] error = new double[this.m_maxIterations + 1];

		SimpleLinearRegression[][] backup = this.m_regressions;

		for (int i = 0; i < m_numFoldsBoosting; i++) {
			// XXX Interrupt added
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}

			// split into training/test data in fold
			Instances train = allData.trainCV(m_numFoldsBoosting, i);
			Instances test = allData.testCV(m_numFoldsBoosting, i);

			// initialize LogitBoost
			this.m_numRegressions = 0;
			this.m_regressions = this.copyRegressions(backup);

			// run LogitBoost iterations
			int iterations = this.performBoosting(train, test, error, completedIterations);
			if (iterations < completedIterations) {
				completedIterations = iterations;
			}
		}

		// determine iteration with minimum error over the folds
		int bestIteration = this.getBestIteration(error, completedIterations);

		// rebuild model on all of the training data
		this.m_numRegressions = 0;
		this.m_regressions = backup;
		this.performBoosting(bestIteration);
	}

	/**
	 * Deep copies the given array of simple linear regression functions.
	 *
	 * @param a
	 *            the array to copy
	 *
	 * @return the new array
	 */
	protected SimpleLinearRegression[][] copyRegressions(final SimpleLinearRegression[][] a) throws Exception {

		SimpleLinearRegression[][] result = this.initRegressions();
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				if (j != this.m_numericDataHeader.classIndex()) {
					result[i][j].addModel(a[i][j]);
				}
			}
		}
		return result;
	}

	/**
	 * Runs LogitBoost, determining the best number of iterations by an information criterion (currently AIC).
	 */
	protected void performBoostingInfCriterion() throws Exception {
		double bestCriterion = Double.MAX_VALUE;
		int bestIteration = 0;
		int noMin = 0;

		// Variable to keep track of criterion values (AIC)
		double criterionValue = Double.MAX_VALUE;

		// initialize Ys/Fs/ps
		double[][] trainYs = this.getYs(this.m_train);
		double[][] trainFs = this.getFs(this.m_numericData);
		double[][] probs = this.getProbs(trainFs);

		int iteration = 0;
		while (iteration < this.m_maxIterations) {
			// XXX interrupt weka
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}

			// perform single LogitBoost iteration
			boolean foundAttribute = this.performIteration(iteration, trainYs, trainFs, probs, this.m_numericData);
			if (foundAttribute) {
				iteration++;
				this.m_numRegressions = iteration;
			} else {
				// could not fit simple linear regression: stop LogitBoost
				break;
			}

			double numberOfAttributes = this.m_numParameters + iteration;

			// Fill criterion array values
			criterionValue = 2.0 * this.negativeLogLikelihood(trainYs, probs) + 2.0 * numberOfAttributes;

			// heuristic: stop LogitBoost if the current minimum has not changed for
			// <m_heuristicStop> iterations
			if (noMin > this.m_heuristicStop) {
				break;
			}
			if (criterionValue < bestCriterion) {
				bestCriterion = criterionValue;
				bestIteration = iteration;
				noMin = 0;
			} else {
				noMin++;
			}
		}

		this.m_numRegressions = 0;
		this.m_regressions = this.initRegressions();
		this.performBoosting(bestIteration);
	}

	/**
	 * Runs LogitBoost on a training set and monitors the error on a test set. Used for running one fold when cross-validating the number of LogitBoost iterations.
	 *
	 * @param train
	 *            the training set
	 * @param test
	 *            the test set
	 * @param error
	 *            array to hold the logged error values
	 * @param maxIterations
	 *            the maximum number of LogitBoost iterations to run
	 * @return the number of completed LogitBoost iterations (can be smaller than maxIterations if the heuristic for early stopping is active or there is a problem while fitting the regressions in LogitBoost).
	 * @throws Exception
	 *             if something goes wrong
	 */
	protected int performBoosting(final Instances train, final Instances test, final double[] error, final int maxIterations) throws Exception {

		// get numeric version of the (sub)set of training instances
		Instances numericTrain = this.getNumericData(train);

		// initialize Ys/Fs/ps
		double[][] trainYs = this.getYs(train);
		double[][] trainFs = this.getFs(numericTrain);
		double[][] probs = this.getProbs(trainFs);

		int iteration = 0;

		int noMin = 0;
		double lastMin = Double.MAX_VALUE;

		if (this.m_errorOnProbabilities) {
			error[0] += this.getMeanAbsoluteError(test);
		} else {
			error[0] += this.getErrorRate(test);
		}

		while (iteration < maxIterations) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}

			// perform single LogitBoost iteration
			boolean foundAttribute = this.performIteration(iteration, trainYs, trainFs, probs, numericTrain);
			if (foundAttribute) {
				iteration++;
				this.m_numRegressions = iteration;
			} else {
				// could not fit simple linear regression: stop LogitBoost
				break;
			}

			if (this.m_errorOnProbabilities) {
				error[iteration] += this.getMeanAbsoluteError(test);
			} else {
				error[iteration] += this.getErrorRate(test);
			}

			// heuristic: stop LogitBoost if the current minimum has not changed for
			// <m_heuristicStop> iterations
			if (noMin > this.m_heuristicStop) {
				break;
			}
			if (error[iteration] < lastMin) {
				lastMin = error[iteration];
				noMin = 0;
			} else {
				noMin++;
			}
		}

		return iteration;
	}

	/**
	 * Runs LogitBoost with a fixed number of iterations.
	 *
	 * @param numIterations
	 *            the number of iterations to run
	 * @throws Exception
	 *             if something goes wrong
	 */
	protected void performBoosting(final int numIterations) throws Exception {

		// initialize Ys/Fs/ps
		double[][] trainYs = this.getYs(this.m_train);
		double[][] trainFs = this.getFs(this.m_numericData);
		double[][] probs = this.getProbs(trainFs);

		int iteration = 0;

		// run iterations
		while (iteration < numIterations) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			boolean foundAttribute = this.performIteration(iteration, trainYs, trainFs, probs, this.m_numericData);
			if (foundAttribute) {
				iteration++;
			} else {
				break;
			}
		}

		this.m_numRegressions = iteration;
	}

	/**
	 * Runs LogitBoost using the stopping criterion on the training set. The number of iterations is used that gives the lowest error on the training set, either misclassification or error on probabilities (depending on the
	 * errorOnProbabilities option).
	 *
	 * @throws Exception
	 *             if something goes wrong
	 */
	protected void performBoosting() throws Exception {

		// initialize Ys/Fs/ps
		double[][] trainYs = this.getYs(this.m_train);
		double[][] trainFs = this.getFs(this.m_numericData);
		double[][] probs = this.getProbs(trainFs);

		int iteration = 0;

		double[] trainErrors = new double[this.m_maxIterations + 1];
		trainErrors[0] = this.getErrorRate(this.m_train);

		int noMin = 0;
		double lastMin = Double.MAX_VALUE;

		while (iteration < this.m_maxIterations) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			boolean foundAttribute = this.performIteration(iteration, trainYs, trainFs, probs, this.m_numericData);
			if (foundAttribute) {
				iteration++;
				this.m_numRegressions = iteration;
			} else {
				// could not fit simple regression
				break;
			}

			trainErrors[iteration] = this.getErrorRate(this.m_train);

			// heuristic: stop LogitBoost if the current minimum has not changed for
			// <m_heuristicStop> iterations
			if (noMin > this.m_heuristicStop) {
				break;
			}
			if (trainErrors[iteration] < lastMin) {
				lastMin = trainErrors[iteration];
				noMin = 0;
			} else {
				noMin++;
			}
		}

		// find iteration with best error
		int bestIteration = this.getBestIteration(trainErrors, iteration);
		this.m_numRegressions = 0;
		this.m_regressions = this.initRegressions();
		this.performBoosting(bestIteration);
	}

	/**
	 * Returns the misclassification error of the current model on a set of instances.
	 *
	 * @param data
	 *            the set of instances
	 * @return the error rate
	 * @throws Exception
	 *             if something goes wrong
	 */
	protected double getErrorRate(final Instances data) throws Exception {
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(this, data);
		return eval.errorRate();
	}

	/**
	 * Returns the error of the probability estimates for the current model on a set of instances.
	 *
	 * @param data
	 *            the set of instances
	 * @return the error
	 * @throws Exception
	 *             if something goes wrong
	 */
	protected double getMeanAbsoluteError(final Instances data) throws Exception {
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(this, data);
		return eval.meanAbsoluteError();
	}

	/**
	 * Helper function to find the minimum in an array of error values.
	 *
	 * @param errors
	 *            an array containing errors
	 * @param maxIteration
	 *            the maximum of iterations
	 * @return the minimum
	 */
	protected int getBestIteration(final double[] errors, final int maxIteration) {
		double bestError = errors[0];
		int bestIteration = 0;
		for (int i = 1; i <= maxIteration; i++) {
			if (errors[i] < bestError) {
				bestError = errors[i];
				bestIteration = i;
			}
		}
		return bestIteration;
	}

	/**
	 * Performs a single iteration of LogitBoost, and updates the model accordingly. A simple regression function is fit to the response and added to the m_regressions array.
	 *
	 * @param iteration
	 *            the current iteration
	 * @param trainYs
	 *            the y-values (see description of LogitBoost) for the model trained so far
	 * @param trainFs
	 *            the F-values (see description of LogitBoost) for the model trained so far
	 * @param probs
	 *            the p-values (see description of LogitBoost) for the model trained so far
	 * @param trainNumeric
	 *            numeric version of the training data
	 * @return returns true if iteration performed successfully, false if no simple regression function could be fitted.
	 * @throws Exception
	 *             if something goes wrong
	 */
	protected boolean performIteration(final int iteration, final double[][] trainYs, final double[][] trainFs, final double[][] probs, final Instances trainNumeric) throws Exception {

		SimpleLinearRegression[] linearRegressionForEachClass = new SimpleLinearRegression[this.m_numClasses];

		// Store weights
		double[] oldWeights = new double[trainNumeric.numInstances()];
		for (int i = 0; i < oldWeights.length; i++) {
			oldWeights[i] = trainNumeric.instance(i).weight();
		}

		for (int j = 0; j < this.m_numClasses; j++) {
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			// Keep track of sum of weights
			double weightSum = 0.0;

			for (int i = 0; i < trainNumeric.numInstances(); i++) {
				if (Thread.interrupted()) {
					throw new InterruptedException("Killed WEKA!");
				}

				// compute response and weight
				double p = probs[i][j];
				double actual = trainYs[i][j];
				double z = this.getZ(actual, p);
				double w = (actual - p) / z;

				// set values for instance
				Instance current = trainNumeric.instance(i);
				current.setValue(trainNumeric.classIndex(), z);
				current.setWeight(oldWeights[i] * w);

				weightSum += current.weight();
			}

			Instances instancesCopy = trainNumeric;

			if (weightSum > 0) {

				// Only the (1-beta)th quantile of instances are sent to the base
				// classifier
				if (this.m_weightTrimBeta > 0) {

					// Need to make an empty dataset
					instancesCopy = new Instances(trainNumeric, trainNumeric.numInstances());

					// Get weights
					double[] weights = new double[oldWeights.length];
					for (int i = 0; i < oldWeights.length; i++) {
						weights[i] = trainNumeric.instance(i).weight();
					}

					double weightPercentage = 0.0;
					int[] weightsOrder = Utils.sort(weights);

					for (int i = weightsOrder.length - 1; (i >= 0) && (weightPercentage < (1 - this.m_weightTrimBeta)); i--) {
						instancesCopy.add(trainNumeric.instance(weightsOrder[i]));
						weightPercentage += (weights[weightsOrder[i]] / weightSum);

					}

					// Update the sum of weights
					weightSum = instancesCopy.sumOfWeights();
				}

				// Scale the weights
				double multiplier = instancesCopy.numInstances() / weightSum;
				for (Instance current : instancesCopy) {
					current.setWeight(current.weight() * multiplier);
				}
			}

			// fit simple regression function
			linearRegressionForEachClass[j] = new SimpleLinearRegression();
			linearRegressionForEachClass[j].buildClassifier(instancesCopy);

			boolean foundAttribute = linearRegressionForEachClass[j].foundUsefulAttribute();
			if (!foundAttribute) {
				// could not fit simple regression function

				// Restore weights
				for (int i = 0; i < oldWeights.length; i++) {
					trainNumeric.instance(i).setWeight(oldWeights[i]);
				}
				return false;
			}
		}

		// Add each linear regression model to the sum
		for (int i = 0; i < this.m_numClasses; i++) {
			this.m_regressions[i][linearRegressionForEachClass[i].getAttributeIndex()].addModel(linearRegressionForEachClass[i]);
		}

		// Evaluate / increment trainFs from the classifier
		for (int i = 0; i < trainFs.length; i++) {
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			double[] pred = new double[this.m_numClasses];
			double predSum = 0;
			for (int j = 0; j < this.m_numClasses; j++) {
				pred[j] = linearRegressionForEachClass[j].classifyInstance(trainNumeric.instance(i));
				predSum += pred[j];
			}
			predSum /= this.m_numClasses;
			for (int j = 0; j < this.m_numClasses; j++) {
				trainFs[i][j] += (pred[j] - predSum) * (this.m_numClasses - 1) / this.m_numClasses;
			}
		}

		// Compute the current probability estimates
		for (int i = 0; i < trainYs.length; i++) {
			probs[i] = this.probs(trainFs[i]);
		}

		// Restore weights
		for (int i = 0; i < oldWeights.length; i++) {
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			trainNumeric.instance(i).setWeight(oldWeights[i]);
		}
		return true;
	}

	/**
	 * Helper function to initialize m_regressions.
	 *
	 * @return the generated classifiers
	 */
	protected SimpleLinearRegression[][] initRegressions() throws Exception {
		SimpleLinearRegression[][] classifiers = new SimpleLinearRegression[this.m_numClasses][this.m_numericDataHeader.numAttributes()];
		for (int j = 0; j < this.m_numClasses; j++) {
			for (int i = 0; i < this.m_numericDataHeader.numAttributes(); i++) {
				if (i != this.m_numericDataHeader.classIndex()) {
					classifiers[j][i] = new SimpleLinearRegression(i, 0, 0);
				}
			}
		}
		return classifiers;
	}

	/**
	 * Private class implementing a DenseInstance with an unsafe setValue() operation.
	 */
	private class UnsafeInstance extends DenseInstance {

		/**
		 * Added ID to avoid warning
		 */
		private static final long serialVersionUID = 3210674215118962869L;

		/**
		 * The constructor.
		 *
		 * @param vals
		 *            The instance whose value we want to copy.
		 */
		public UnsafeInstance(final Instance vals) {

			super(vals.numAttributes());
			for (int i = 0; i < vals.numAttributes(); i++) {
				this.m_AttValues[i] = vals.value(i);
			}
			this.m_Weight = vals.weight();
		}

		/**
		 * Unsafe setValue() method.
		 */
		@Override
		public void setValue(final int attIndex, final double value) {

			this.m_AttValues[attIndex] = value;
		}

		/**
		 * We need a copy method that doesn't do anything...
		 */
		@Override
		public Object copy() {

			return this;
		}
	}

	/**
	 * Converts training data to numeric version. The class variable is replaced by a pseudo-class used by LogitBoost.
	 *
	 * @param data
	 *            the data to convert
	 * @return the converted data
	 * @throws Exception
	 *             if something goes wrong
	 */
	protected Instances getNumericData(final Instances data) throws Exception {

		if (this.m_numericDataHeader == null) {
			this.m_numericDataHeader = new Instances(data, 0);

			int classIndex = this.m_numericDataHeader.classIndex();
			this.m_numericDataHeader.setClassIndex(-1);
			this.m_numericDataHeader.replaceAttributeAt(new Attribute("'pseudo class'"), classIndex);
			this.m_numericDataHeader.setClassIndex(classIndex);
		}
		Instances numericData = new Instances(this.m_numericDataHeader, data.numInstances());
		for (Instance inst : data) {
			// XXX Interrupt added
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			numericData.add(new UnsafeInstance(inst));
		}

		return numericData;
	}

	/**
	 * Computes the LogitBoost response variable from y/p values (actual/estimated class probabilities).
	 *
	 * @param actual
	 *            the actual class probability
	 * @param p
	 *            the estimated class probability
	 * @return the LogitBoost response
	 */
	protected double getZ(final double actual, final double p) {
		double z;
		if (actual == 1) {
			z = 1.0 / p;
			if (z > Z_MAX) { // threshold
				z = Z_MAX;
			}
		} else {
			z = -1.0 / (1.0 - p);
			if (z < -Z_MAX) { // threshold
				z = -Z_MAX;
			}
		}
		return z;
	}

	/**
	 * Computes the LogitBoost response for an array of y/p values (actual/estimated class probabilities).
	 *
	 * @param dataYs
	 *            the actual class probabilities
	 * @param probs
	 *            the estimated class probabilities
	 * @return the LogitBoost response
	 * @throws InterruptedException
	 */
	protected double[][] getZs(final double[][] probs, final double[][] dataYs) throws InterruptedException {

		double[][] dataZs = new double[probs.length][this.m_numClasses];
		for (int j = 0; j < this.m_numClasses; j++) {
			// XXX Interrupt added
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			for (int i = 0; i < probs.length; i++) {
				dataZs[i][j] = this.getZ(dataYs[i][j], probs[i][j]);
			}
		}
		return dataZs;
	}

	/**
	 * Computes the LogitBoost weights from an array of y/p values (actual/estimated class probabilities).
	 *
	 * @param dataYs
	 *            the actual class probabilities
	 * @param probs
	 *            the estimated class probabilities
	 * @return the LogitBoost weights
	 */
	protected double[][] getWs(final double[][] probs, final double[][] dataYs) {

		double[][] dataWs = new double[probs.length][this.m_numClasses];
		for (int j = 0; j < this.m_numClasses; j++) {
			for (int i = 0; i < probs.length; i++) {
				double z = this.getZ(dataYs[i][j], probs[i][j]);
				dataWs[i][j] = (dataYs[i][j] - probs[i][j]) / z;
			}
		}
		return dataWs;
	}

	/**
	 * Computes the p-values (probabilities for the classes) from the F-values of the logistic model.
	 *
	 * @param Fs
	 *            the F-values
	 * @return the p-values
	 */
	protected double[] probs(final double[] Fs) {

		double maxF = -Double.MAX_VALUE;
		for (double element : Fs) {
			if (element > maxF) {
				maxF = element;
			}
		}
		double sum = 0;
		double[] probs = new double[Fs.length];
		for (int i = 0; i < Fs.length; i++) {
			probs[i] = Math.exp(Fs[i] - maxF);
			sum += probs[i];
		}

		Utils.normalize(probs, sum);
		return probs;
	}

	/**
	 * Computes the Y-values (actual class probabilities) for a set of instances.
	 *
	 * @param data
	 *            the data to compute the Y-values from
	 * @return the Y-values
	 */
	protected double[][] getYs(final Instances data) {

		double[][] dataYs = new double[data.numInstances()][this.m_numClasses];
		for (int j = 0; j < this.m_numClasses; j++) {
			for (int k = 0; k < data.numInstances(); k++) {
				dataYs[k][j] = (data.instance(k).classValue() == j) ? 1.0 : 0.0;
			}
		}
		return dataYs;
	}

	/**
	 * Computes the F-values for a single instance.
	 *
	 * @param instance
	 *            the instance to compute the F-values for
	 * @return the F-values
	 * @throws Exception
	 *             if something goes wrong
	 */
	protected double[] getFs(final Instance instance) throws Exception {

		double[] pred = new double[this.m_numClasses];
		double[] instanceFs = new double[this.m_numClasses];

		// add up the predictions from the simple regression functions
		for (int i = 0; i < this.m_numericDataHeader.numAttributes(); i++) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}

			if (i != this.m_numericDataHeader.classIndex()) {
				double predSum = 0;
				for (int j = 0; j < this.m_numClasses; j++) {
					pred[j] = this.m_regressions[j][i].classifyInstance(instance);
					predSum += pred[j];
				}
				predSum /= this.m_numClasses;
				for (int j = 0; j < this.m_numClasses; j++) {
					instanceFs[j] += (pred[j] - predSum) * (this.m_numClasses - 1) / this.m_numClasses;
				}
			}
		}

		return instanceFs;
	}

	/**
	 * Computes the F-values for a set of instances.
	 *
	 * @param data
	 *            the data to work on
	 * @return the F-values
	 * @throws Exception
	 *             if something goes wrong
	 */
	protected double[][] getFs(final Instances data) throws Exception {

		double[][] dataFs = new double[data.numInstances()][];

		for (int k = 0; k < data.numInstances(); k++) {
			// XXX Interrupt added
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			dataFs[k] = this.getFs(data.instance(k));
		}

		return dataFs;
	}

	/**
	 * Computes the p-values (probabilities for the different classes) from the F-values for a set of instances.
	 *
	 * @param dataFs
	 *            the F-values
	 * @return the p-values
	 */
	protected double[][] getProbs(final double[][] dataFs) {

		int numInstances = dataFs.length;
		double[][] probs = new double[numInstances][];

		for (int k = 0; k < numInstances; k++) {
			probs[k] = this.probs(dataFs[k]);
		}
		return probs;
	}

	/**
	 * Returns the negative loglikelihood of the Y-values (actual class probabilities) given the p-values (current probability estimates).
	 *
	 * @param dataYs
	 *            the Y-values
	 * @param probs
	 *            the p-values
	 * @return the likelihood
	 * @throws InterruptedException
	 */
	protected double negativeLogLikelihood(final double[][] dataYs, final double[][] probs) throws InterruptedException {

		double logLikelihood = 0;
		for (int i = 0; i < dataYs.length; i++) {
			for (int j = 0; j < this.m_numClasses; j++) {
				// XXX Interrupt added
				if (Thread.interrupted()) {
					throw new InterruptedException("Killed WEKA!");
				}
				if (dataYs[i][j] == 1.0) {
					logLikelihood -= Math.log(probs[i][j]);
				}
			}
		}
		return logLikelihood;// / (double)dataYs.length;
	}

	/**
	 * Returns an array of the indices of the attributes used in the logistic model. The first dimension is the class, the second dimension holds a list of attribute indices. Attribute indices start at zero.
	 *
	 * @return the array of attribute indices
	 */
	public int[][] getUsedAttributes() {

		int[][] usedAttributes = new int[this.m_numClasses][];

		// first extract coefficients
		double[][] coefficients = this.getCoefficients();

		for (int j = 0; j < this.m_numClasses; j++) {

			// boolean array indicating if attribute used
			boolean[] attributes = new boolean[this.m_numericDataHeader.numAttributes()];
			for (int i = 0; i < attributes.length; i++) {
				// attribute used if coefficient > 0
				if (!Utils.eq(coefficients[j][i + 1], 0)) {
					attributes[i] = true;
				}
			}

			int numAttributes = 0;
			for (int i = 0; i < this.m_numericDataHeader.numAttributes(); i++) {
				if (attributes[i]) {
					numAttributes++;
				}
			}

			// "collect" all attributes into array of indices
			int[] usedAttributesClass = new int[numAttributes];
			int count = 0;
			for (int i = 0; i < this.m_numericDataHeader.numAttributes(); i++) {
				if (attributes[i]) {
					usedAttributesClass[count] = i;
					count++;
				}
			}

			usedAttributes[j] = usedAttributesClass;
		}

		return usedAttributes;
	}

	/**
	 * The number of LogitBoost iterations performed (= the number of simple regression functions fit).
	 *
	 * @return the number of LogitBoost iterations performed
	 */
	public int getNumRegressions() {
		return this.m_numRegressions;
	}

	/**
	 * Get the value of weightTrimBeta.
	 *
	 * @return Value of weightTrimBeta.
	 */
	public double getWeightTrimBeta() {
		return this.m_weightTrimBeta;
	}

	/**
	 * Get the value of useAIC.
	 *
	 * @return Value of useAIC.
	 */
	public boolean getUseAIC() {
		return this.m_useAIC;
	}

	/**
	 * Sets the parameter "maxIterations".
	 *
	 * @param maxIterations
	 *            the maximum iterations
	 */
	public void setMaxIterations(final int maxIterations) {
		this.m_maxIterations = maxIterations;
	}

	/**
	 * Sets the option "heuristicStop".
	 *
	 * @param heuristicStop
	 *            the heuristic stop to use
	 */
	public void setHeuristicStop(final int heuristicStop) {
		this.m_heuristicStop = heuristicStop;
	}

	/**
	 * Sets the option "weightTrimBeta".
	 */
	public void setWeightTrimBeta(final double w) {
		this.m_weightTrimBeta = w;
	}

	/**
	 * Set the value of useAIC.
	 *
	 * @param c
	 *            Value to assign to useAIC.
	 */
	public void setUseAIC(final boolean c) {
		this.m_useAIC = c;
	}

	/**
	 * Returns the maxIterations parameter.
	 *
	 * @return the maximum iteration
	 */
	public int getMaxIterations() {
		return this.m_maxIterations;
	}

	/**
	 * Returns an array holding the coefficients of the logistic model. First dimension is the class, the second one holds a list of coefficients. At position zero, the constant term of the model is stored, then, the coefficients for the
	 * attributes in ascending order.
	 *
	 * @return the array of coefficients
	 */
	protected double[][] getCoefficients() {
		double[][] coefficients = new double[this.m_numClasses][this.m_numericDataHeader.numAttributes() + 1];
		for (int j = 0; j < this.m_numClasses; j++) {
			// go through simple regression functions and add their coefficient to the
			// coefficient of
			// the attribute they are built on.
			for (int i = 0; i < this.m_numericDataHeader.numAttributes(); i++) {
				if (i != this.m_numericDataHeader.classIndex()) {
					double slope = this.m_regressions[j][i].getSlope();
					double intercept = this.m_regressions[j][i].getIntercept();
					int attribute = this.m_regressions[j][i].getAttributeIndex();

					coefficients[j][0] += intercept;
					coefficients[j][attribute + 1] += slope;
				}
			}
		}

		// Need to multiply all coefficients by (J-1) / J
		for (int j = 0; j < coefficients.length; j++) {
			for (int i = 0; i < coefficients[0].length; i++) {
				coefficients[j][i] *= (double) (this.m_numClasses - 1) / (double) this.m_numClasses;
			}
		}

		return coefficients;
	}

	/**
	 * Returns the fraction of all attributes in the data that are used in the logistic model (in percent). An attribute is used in the model if it is used in any of the models for the different classes.
	 *
	 * @return the fraction of all attributes that are used
	 */
	public double percentAttributesUsed() {
		boolean[] attributes = new boolean[this.m_numericDataHeader.numAttributes()];

		double[][] coefficients = this.getCoefficients();
		for (int j = 0; j < this.m_numClasses; j++) {
			for (int i = 1; i < this.m_numericDataHeader.numAttributes() + 1; i++) {
				// attribute used if it is used in any class, note coefficients are
				// shifted by one (because
				// of constant term).
				if (!Utils.eq(coefficients[j][i], 0)) {
					attributes[i - 1] = true;
				}
			}
		}

		// count number of used attributes (without the class attribute)
		double count = 0;
		for (boolean attribute : attributes) {
			if (attribute) {
				count++;
			}
		}
		return count / (this.m_numericDataHeader.numAttributes() - 1) * 100.0;
	}

	/**
	 * Returns a description of the logistic model (i.e., attributes and coefficients).
	 *
	 * @return the description of the model
	 */
	@Override
	public String toString() {

		StringBuffer s = new StringBuffer();

		// get used attributes
		int[][] attributes = this.getUsedAttributes();

		// get coefficients
		double[][] coefficients = this.getCoefficients();

		for (int j = 0; j < this.m_numClasses; j++) {
			s.append("\nClass " + this.m_train.classAttribute().value(j) + " :\n");
			// constant term
			s.append(Utils.doubleToString(coefficients[j][0], 2 + this.m_numDecimalPlaces, this.m_numDecimalPlaces) + " + \n");
			for (int i = 0; i < attributes[j].length; i++) {
				// attribute/coefficient pairs
				s.append("[" + this.m_numericDataHeader.attribute(attributes[j][i]).name() + "]");
				s.append(" * " + Utils.doubleToString(coefficients[j][attributes[j][i] + 1], 2 + this.m_numDecimalPlaces, this.m_numDecimalPlaces));
				if (i != attributes[j].length - 1) {
					s.append(" +");
				}
				s.append("\n");
			}
		}
		return new String(s);
	}

	/**
	 * Returns class probabilities for an instance.
	 *
	 * @param instance
	 *            the instance to compute the distribution for
	 * @return the class probabilities
	 * @throws Exception
	 *             if distribution can't be computed successfully
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		instance = (Instance) instance.copy();

		// set to numeric pseudo-class
		instance.setDataset(this.m_numericDataHeader);

		// calculate probs via Fs
		return this.probs(this.getFs(instance));
	}

	/**
	 * Cleanup in order to save memory.
	 */
	public void cleanup() {
		// save just header info
		this.m_train = new Instances(this.m_train, 0);
		this.m_numericData = null;
	}

	/**
	 * Returns the revision string.
	 *
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision$");
	}
}
