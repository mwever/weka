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
 *    ClassifierSplitModel.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.j48;

import java.io.Serializable;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.Utils;

/**
 * Abstract class for classification models that can be used
 * recursively to split the data.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */
public abstract class ClassifierSplitModel implements Cloneable, Serializable, RevisionHandler {

	/** for serialization */
	private static final long serialVersionUID = 4280730118393457457L;

	/** Distribution of class values. */
	protected Distribution m_distribution;

	/** Number of created subsets. */
	protected int m_numSubsets;

	/**
	 * Allows to clone a model (shallow copy).
	 */
	@Override
	public Object clone() {

		Object clone = null;

		try {
			clone = super.clone();
		} catch (CloneNotSupportedException e) {
		}
		return clone;
	}

	/**
	 * Builds the classifier split model for the given set of instances.
	 *
	 * @exception Exception if something goes wrong
	 */
	public abstract void buildClassifier(Instances instances) throws Exception;

	/**
	 * Checks if generated model is valid.
	 */
	public final boolean checkModel() {

		if (this.m_numSubsets > 0) {
			return true;
		} else {
			return false;
		}
	}

	/**
	 * Classifies a given instance.
	 *
	 * @exception Exception if something goes wrong
	 */
	public final double classifyInstance(final Instance instance) throws Exception {

		int theSubset;

		theSubset = this.whichSubset(instance);
		if (theSubset > -1) {
			return this.m_distribution.maxClass(theSubset);
		} else {
			return this.m_distribution.maxClass();
		}
	}

	/**
	 * Gets class probability for instance.
	 *
	 * @exception Exception if something goes wrong
	 */
	public double classProb(final int classIndex, final Instance instance, final int theSubset) throws Exception {

		if (theSubset > -1) {
			return this.m_distribution.prob(classIndex, theSubset);
		} else {
			double[] weights = this.weights(instance);
			if (weights == null) {
				return this.m_distribution.prob(classIndex);
			} else {
				double prob = 0;
				for (int i = 0; i < weights.length; i++) {
					prob += weights[i] * this.m_distribution.prob(classIndex, i);
				}
				return prob;
			}
		}
	}

	/**
	 * Gets class probability for instance.
	 *
	 * @exception Exception if something goes wrong
	 */
	public double classProbLaplace(final int classIndex, final Instance instance, final int theSubset) throws Exception {

		if (theSubset > -1) {
			return this.m_distribution.laplaceProb(classIndex, theSubset);
		} else {
			double[] weights = this.weights(instance);
			if (weights == null) {
				return this.m_distribution.laplaceProb(classIndex);
			} else {
				double prob = 0;
				for (int i = 0; i < weights.length; i++) {
					prob += weights[i] * this.m_distribution.laplaceProb(classIndex, i);
				}
				return prob;
			}
		}
	}

	/**
	 * Returns coding costs of model. Returns 0 if not overwritten.
	 */
	public double codingCost() {

		return 0;
	}

	/**
	 * Returns the distribution of class values induced by the model.
	 */
	public final Distribution distribution() {

		return this.m_distribution;
	}

	/**
	 * Prints left side of condition satisfied by instances.
	 *
	 * @param data the data.
	 */
	public abstract String leftSide(Instances data);

	/**
	 * Prints left side of condition satisfied by instances in subset index.
	 */
	public abstract String rightSide(int index, Instances data);

	/**
	 * Prints label for subset index of instances (eg class).
	 *
	 * @exception Exception if something goes wrong
	 */
	public final String dumpLabel(final int index, final Instances data) throws Exception {

		StringBuffer text;

		text = new StringBuffer();
		text.append(data.classAttribute().value(this.m_distribution.maxClass(index)));
		text.append(" (" + Utils.roundDouble(this.m_distribution.perBag(index), 2));
		if (Utils.gr(this.m_distribution.numIncorrect(index), 0)) {
			text.append("/" + Utils.roundDouble(this.m_distribution.numIncorrect(index), 2));
		}
		text.append(")");

		return text.toString();
	}

	public final String sourceClass(final int index, final Instances data) throws Exception {

		System.err.println("sourceClass");
		return (new StringBuffer(this.m_distribution.maxClass(index))).toString();
	}

	public abstract String sourceExpression(int index, Instances data);

	/**
	 * Prints the split model.
	 *
	 * @exception Exception if something goes wrong
	 */
	public final String dumpModel(final Instances data) throws Exception {

		StringBuffer text;
		int i;

		text = new StringBuffer();
		for (i = 0; i < this.m_numSubsets; i++) {
			text.append(this.leftSide(data) + this.rightSide(i, data) + ": ");
			text.append(this.dumpLabel(i, data) + "\n");
		}
		return text.toString();
	}

	/**
	 * Returns the number of created subsets for the split.
	 */
	public final int numSubsets() {

		return this.m_numSubsets;
	}

	/**
	 * Sets distribution associated with model.
	 */
	public void resetDistribution(final Instances data) throws Exception {

		this.m_distribution = new Distribution(data, this);
	}

	/**
	 * Sets the distribution associated with model.
	 *
	 * @param dist
	 */
	public void setDistribution(final Distribution dist) {
		this.m_distribution = dist;
	}

	/**
	 * Splits the given set of instances into subsets.
	 *
	 * @exception Exception if something goes wrong
	 */
	public final Instances[] split(final Instances data) throws Exception {

		// Find size and constitution of subsets
		int[] subsetSize = new int[this.m_numSubsets];
		for (Instance instance : data) {
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			int subset = this.whichSubset(instance);
			if (subset > -1) {
				subsetSize[subset]++;
			} else {
				double[] weights = this.weights(instance);
				for (int j = 0; j < this.m_numSubsets; j++) {
					if (Utils.gr(weights[j], 0)) {
						subsetSize[j]++;
					}
				}
			}
		}

		// Create subsets
		Instances[] instances = new Instances[this.m_numSubsets];
		for (int j = 0; j < this.m_numSubsets; j++) {
			instances[j] = new Instances(data, subsetSize[j]);
		}
		for (Instance instance : data) {
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			int subset = this.whichSubset(instance);
			if (subset > -1) {
				instances[subset].add(instance);
			} else {
				double[] weights = this.weights(instance);
				for (int j = 0; j < this.m_numSubsets; j++) {
					if (Utils.gr(weights[j], 0)) {
						instances[j].add(instance);
						instances[j].lastInstance().setWeight(weights[j] * instance.weight());
					}
				}
			}
		}

		return instances;
	}

	/**
	 * Returns weights if instance is assigned to more than one subset.
	 * Returns null if instance is only assigned to one subset.
	 */
	public abstract double[] weights(Instance instance);

	/**
	 * Returns index of subset instance is assigned to.
	 * Returns -1 if instance is assigned to more than one subset.
	 *
	 * @exception Exception if something goes wrong
	 */
	public abstract int whichSubset(Instance instance) throws Exception;
}
