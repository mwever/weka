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
 *    NBTreeSplit.java
 *    Copyright (C) 2004-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.j48;

import java.util.Random;

import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/**
 * Class implementing a NBTree split on an attribute.
 *
 * @author Mark Hall (mhall@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class NBTreeSplit extends ClassifierSplitModel {

	/** for serialization */
	private static final long serialVersionUID = 8922627123884975070L;

	/** Desired number of branches. */
	protected int m_complexityIndex;

	/** Attribute to split on. */
	protected final int m_attIndex;

	/** The sum of the weights of the instances. */
	protected final double m_sumOfWeights;

	/**
	 * The weight of the instances incorrectly classified by the naive bayes
	 * models arising from this split
	 */
	protected double m_errors;

	protected C45Split m_c45S;

	/** The global naive bayes model for this node */
	NBTreeNoSplit m_globalNB;

	/**
	 * Initializes the split model.
	 */
	public NBTreeSplit(final int attIndex, final int minNoObj, final double sumOfWeights) {

		// Get index of attribute to split on.
		this.m_attIndex = attIndex;

		// Set the sum of the weights
		this.m_sumOfWeights = sumOfWeights;

	}

	/**
	 * Creates a NBTree-type split on the given data. Assumes that none of the
	 * class values is missing.
	 * 
	 * @exception Exception if something goes wrong
	 */
	@Override
	public void buildClassifier(final Instances trainInstances) throws Exception {

		// Initialize the remaining instance variables.
		this.m_numSubsets = 0;
		this.m_errors = 0;
		if (this.m_globalNB != null) {
			this.m_errors = this.m_globalNB.getErrors();
		}

		// Different treatment for enumerated and numeric
		// attributes.
		if (trainInstances.attribute(this.m_attIndex).isNominal()) {
			this.m_complexityIndex = trainInstances.attribute(this.m_attIndex).numValues();
			this.handleEnumeratedAttribute(trainInstances);
		} else {
			this.m_complexityIndex = 2;
			trainInstances.sort(trainInstances.attribute(this.m_attIndex));
			this.handleNumericAttribute(trainInstances);
		}
	}

	/**
	 * Returns index of attribute for which split was generated.
	 */
	public final int attIndex() {

		return this.m_attIndex;
	}

	/**
	 * Creates split on enumerated attribute.
	 * 
	 * @exception Exception if something goes wrong
	 */
	private void handleEnumeratedAttribute(final Instances trainInstances) throws Exception {

		this.m_c45S = new C45Split(this.m_attIndex, 2, this.m_sumOfWeights, true);
		this.m_c45S.buildClassifier(trainInstances);
		if (this.m_c45S.numSubsets() == 0) {
			return;
		}
		this.m_errors = 0;
		Instance instance;

		Instances[] trainingSets = new Instances[this.m_complexityIndex];
		for (int i = 0; i < this.m_complexityIndex; i++) {
			trainingSets[i] = new Instances(trainInstances, 0);
		}
		/*
		 * m_distribution = new Distribution(m_complexityIndex,
		 * trainInstances.numClasses());
		 */
		int subset;
		for (int i = 0; i < trainInstances.numInstances(); i++) {
			instance = trainInstances.instance(i);
			subset = this.m_c45S.whichSubset(instance);
			if (subset > -1) {
				trainingSets[subset].add((Instance) instance.copy());
			} else {
				double[] weights = this.m_c45S.weights(instance);
				for (int j = 0; j < this.m_complexityIndex; j++) {
					try {
						Instance temp = (Instance) instance.copy();
						if (weights.length == this.m_complexityIndex) {
							temp.setWeight(temp.weight() * weights[j]);
						} else {
							temp.setWeight(temp.weight() / this.m_complexityIndex);
						}
						trainingSets[j].add(temp);
					} catch (Exception ex) {
						System.err.println("*** " + this.m_complexityIndex);
						System.err.println(weights.length);
						throw ex;
					}
				}
			}
		}

		/*
		 * // compute weights (weights of instances per subset m_weights = new
		 * double [m_complexityIndex]; for (int i = 0; i < m_complexityIndex; i++) {
		 * m_weights[i] = trainingSets[i].sumOfWeights(); }
		 * Utils.normalize(m_weights);
		 */

		/*
		 * // Only Instances with known values are relevant. Enumeration enu =
		 * trainInstances.enumerateInstances(); while (enu.hasMoreElements()) {
		 * instance = (Instance) enu.nextElement(); if
		 * (!instance.isMissing(m_attIndex)) { //
		 * m_distribution.add((int)instance.value(m_attIndex),instance);
		 * trainingSets[(int)instances.value(m_attIndex)].add(instance); } else { //
		 * add these to the error count m_errors += instance.weight(); } }
		 */

		Random r = new Random(1);
		int minNumCount = 0;
		for (int i = 0; i < this.m_complexityIndex; i++) {
			if (trainingSets[i].numInstances() >= 5) {
				minNumCount++;
				// Discretize the sets
				Discretize disc = new Discretize();
				disc.setInputFormat(trainingSets[i]);
				trainingSets[i] = Filter.useFilter(trainingSets[i], disc);

				trainingSets[i].randomize(r);
				trainingSets[i].stratify(5);
				NaiveBayesUpdateable fullModel = new NaiveBayesUpdateable();
				fullModel.buildClassifier(trainingSets[i]);

				// add the errors for this branch of the split
				this.m_errors += NBTreeNoSplit.crossValidate(fullModel, trainingSets[i], r);
			} else {
				// if fewer than min obj then just count them as errors
				for (int j = 0; j < trainingSets[i].numInstances(); j++) {
					this.m_errors += trainingSets[i].instance(j).weight();
				}
			}
		}

		// Check if there are at least five instances in at least two of the subsets
		// subsets.
		if (minNumCount > 1) {
			this.m_numSubsets = this.m_complexityIndex;
		}
	}

	/**
	 * Creates split on numeric attribute.
	 * 
	 * @exception Exception if something goes wrong
	 */
	private void handleNumericAttribute(final Instances trainInstances) throws Exception {

		this.m_c45S = new C45Split(this.m_attIndex, 2, this.m_sumOfWeights, true);
		this.m_c45S.buildClassifier(trainInstances);
		if (this.m_c45S.numSubsets() == 0) {
			return;
		}
		this.m_errors = 0;

		Instances[] trainingSets = new Instances[this.m_complexityIndex];
		trainingSets[0] = new Instances(trainInstances, 0);
		trainingSets[1] = new Instances(trainInstances, 0);
		int subset = -1;

		// populate the subsets
		for (int i = 0; i < trainInstances.numInstances(); i++) {
			Instance instance = trainInstances.instance(i);
			subset = this.m_c45S.whichSubset(instance);
			if (subset != -1) {
				trainingSets[subset].add((Instance) instance.copy());
			} else {
				double[] weights = this.m_c45S.weights(instance);
				for (int j = 0; j < this.m_complexityIndex; j++) {
					Instance temp = (Instance) instance.copy();
					if (weights.length == this.m_complexityIndex) {
						temp.setWeight(temp.weight() * weights[j]);
					} else {
						temp.setWeight(temp.weight() / this.m_complexityIndex);
					}
					trainingSets[j].add(temp);
				}
			}
		}

		/*
		 * // compute weights (weights of instances per subset m_weights = new
		 * double [m_complexityIndex]; for (int i = 0; i < m_complexityIndex; i++) {
		 * m_weights[i] = trainingSets[i].sumOfWeights(); }
		 * Utils.normalize(m_weights);
		 */

		Random r = new Random(1);
		int minNumCount = 0;
		for (int i = 0; i < this.m_complexityIndex; i++) {
			if (trainingSets[i].numInstances() > 5) {
				minNumCount++;
				// Discretize the sets
				Discretize disc = new Discretize();
				disc.setInputFormat(trainingSets[i]);
				trainingSets[i] = Filter.useFilter(trainingSets[i], disc);

				trainingSets[i].randomize(r);
				trainingSets[i].stratify(5);
				NaiveBayesUpdateable fullModel = new NaiveBayesUpdateable();
				fullModel.buildClassifier(trainingSets[i]);

				// add the errors for this branch of the split
				this.m_errors += NBTreeNoSplit.crossValidate(fullModel, trainingSets[i], r);
			} else {
				for (int j = 0; j < trainingSets[i].numInstances(); j++) {
					this.m_errors += trainingSets[i].instance(j).weight();
				}
			}
		}

		// Check if minimum number of Instances in at least two
		// subsets.
		if (minNumCount > 1) {
			this.m_numSubsets = this.m_complexityIndex;
		}
	}

	/**
	 * Returns index of subset instance is assigned to. Returns -1 if instance is
	 * assigned to more than one subset.
	 * 
	 * @exception Exception if something goes wrong
	 */
	@Override
	public final int whichSubset(final Instance instance) throws Exception {

		return this.m_c45S.whichSubset(instance);
	}

	/**
	 * Returns weights if instance is assigned to more than one subset. Returns
	 * null if instance is only assigned to one subset.
	 */
	@Override
	public final double[] weights(final Instance instance) {
		return this.m_c45S.weights(instance);
		// return m_weights;
	}

	/**
	 * Returns a string containing java source code equivalent to the test made at
	 * this node. The instance being tested is called "i".
	 * 
	 * @param index index of the nominal value tested
	 * @param data the data containing instance structure info
	 * @return a value of type 'String'
	 */
	@Override
	public final String sourceExpression(final int index, final Instances data) {
		return this.m_c45S.sourceExpression(index, data);
	}

	/**
	 * Prints the condition satisfied by instances in a subset.
	 * 
	 * @param index of subset
	 * @param data training set.
	 */
	@Override
	public final String rightSide(final int index, final Instances data) {
		return this.m_c45S.rightSide(index, data);
	}

	/**
	 * Prints left side of condition..
	 * 
	 * @param data training set.
	 */
	@Override
	public final String leftSide(final Instances data) {

		return this.m_c45S.leftSide(data);
	}

	/**
	 * Return the probability for a class value
	 * 
	 * @param classIndex the index of the class value
	 * @param instance the instance to generate a probability for
	 * @param theSubset the subset to consider
	 * @return a probability
	 * @exception Exception if an error occurs
	 */
	@Override
	public double classProb(final int classIndex, final Instance instance, final int theSubset) throws Exception {

		// use the global naive bayes model
		if (theSubset > -1) {
			return this.m_globalNB.classProb(classIndex, instance, theSubset);
		} else {
			throw new Exception("This shouldn't happen!!!");
		}
	}

	/**
	 * Return the global naive bayes model for this node
	 * 
	 * @return a <code>NBTreeNoSplit</code> value
	 */
	public NBTreeNoSplit getGlobalModel() {
		return this.m_globalNB;
	}

	/**
	 * Set the global naive bayes model for this node
	 * 
	 * @param global a <code>NBTreeNoSplit</code> value
	 */
	public void setGlobalModel(final NBTreeNoSplit global) {
		this.m_globalNB = global;
	}

	/**
	 * Return the errors made by the naive bayes models arising from this split.
	 * 
	 * @return a <code>double</code> value
	 */
	public double getErrors() {
		return this.m_errors;
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
