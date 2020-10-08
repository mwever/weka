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
 *    C45Split.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.j48;

import java.util.Enumeration;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * Class implementing a C4.5-type split on an attribute.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class C45Split extends ClassifierSplitModel {

	/** for serialization */
	private static final long serialVersionUID = 3064079330067903161L;

	/** Desired number of branches. */
	protected int m_complexityIndex;

	/** Attribute to split on. */
	protected final int m_attIndex;

	/** Minimum number of objects in a split. */
	protected final int m_minNoObj;

	/** Use MDL correction? */
	protected final boolean m_useMDLcorrection;

	/** Value of split point. */
	protected double m_splitPoint;

	/** InfoGain of split. */
	protected double m_infoGain;

	/** GainRatio of split. */
	protected double m_gainRatio;

	/** The sum of the weights of the instances. */
	protected final double m_sumOfWeights;

	/** Number of split points. */
	protected int m_index;

	/** Static reference to splitting criterion. */
	protected static InfoGainSplitCrit infoGainCrit = new InfoGainSplitCrit();

	/** Static reference to splitting criterion. */
	protected static GainRatioSplitCrit gainRatioCrit = new GainRatioSplitCrit();

	/**
	 * Initializes the split model.
	 */
	public C45Split(final int attIndex, final int minNoObj, final double sumOfWeights, final boolean useMDLcorrection) {

		// Get index of attribute to split on.
		this.m_attIndex = attIndex;

		// Set minimum number of objects.
		this.m_minNoObj = minNoObj;

		// Set the sum of the weights
		this.m_sumOfWeights = sumOfWeights;

		// Whether to use the MDL correction for numeric attributes
		this.m_useMDLcorrection = useMDLcorrection;
	}

	/**
	 * Creates a C4.5-type split on the given data. Assumes that none of the class values is missing.
	 *
	 * @exception Exception
	 *              if something goes wrong
	 */
	@Override
	public void buildClassifier(final Instances trainInstances) throws Exception {

		// Initialize the remaining instance variables.
		this.m_numSubsets = 0;
		this.m_splitPoint = Double.MAX_VALUE;
		this.m_infoGain = 0;
		this.m_gainRatio = 0;

		// Different treatment for enumerated and numeric
		// attributes.
		if (trainInstances.attribute(this.m_attIndex).isNominal()) {
			this.m_complexityIndex = trainInstances.attribute(this.m_attIndex).numValues();
			this.m_index = this.m_complexityIndex;
			this.handleEnumeratedAttribute(trainInstances);
		} else {
			this.m_complexityIndex = 2;
			this.m_index = 0;
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
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
	 * Returns the split point (numeric attribute only).
	 *
	 * @return the split point used for a test on a numeric attribute
	 */
	public double splitPoint() {
		return this.m_splitPoint;
	}

	/**
	 * Gets class probability for instance.
	 *
	 * @exception Exception
	 *              if something goes wrong
	 */
	@Override
	public final double classProb(final int classIndex, final Instance instance, final int theSubset) throws Exception {

		if (theSubset <= -1) {
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
		} else {
			if (Utils.gr(this.m_distribution.perBag(theSubset), 0)) {
				return this.m_distribution.prob(classIndex, theSubset);
			} else {
				return this.m_distribution.prob(classIndex);
			}
		}
	}

	/**
	 * Returns coding cost for split (used in rule learner).
	 */
	@Override
	public final double codingCost() {

		return Utils.log2(this.m_index);
	}

	/**
	 * Returns (C4.5-type) gain ratio for the generated split.
	 */
	public final double gainRatio() {
		return this.m_gainRatio;
	}

	/**
	 * Creates split on enumerated attribute.
	 *
	 * @exception Exception
	 *              if something goes wrong
	 */
	private void handleEnumeratedAttribute(final Instances trainInstances) throws Exception {

		Instance instance;

		this.m_distribution = new Distribution(this.m_complexityIndex, trainInstances.numClasses());

		// Only Instances with known values are relevant.
		Enumeration<Instance> enu = trainInstances.enumerateInstances();
		while (enu.hasMoreElements()) {
			instance = enu.nextElement();
			if (!instance.isMissing(this.m_attIndex)) {
				this.m_distribution.add((int) instance.value(this.m_attIndex), instance);
			}
		}

		// Check if minimum number of Instances in at least two
		// subsets.
		if (this.m_distribution.check(this.m_minNoObj)) {
			this.m_numSubsets = this.m_complexityIndex;
			this.m_infoGain = infoGainCrit.splitCritValue(this.m_distribution, this.m_sumOfWeights);
			this.m_gainRatio = gainRatioCrit.splitCritValue(this.m_distribution, this.m_sumOfWeights, this.m_infoGain);
		}
	}

	/**
	 * Creates split on numeric attribute.
	 *
	 * @exception Exception
	 *              if something goes wrong
	 */
	private void handleNumericAttribute(final Instances trainInstances) throws Exception {

		int firstMiss;
		int next = 1;
		int last = 0;
		int splitIndex = -1;
		double currentInfoGain;
		double defaultEnt;
		double minSplit;
		Instance instance;
		int i;

		// Current attribute is a numeric attribute.
		this.m_distribution = new Distribution(2, trainInstances.numClasses());

		// Only Instances with known values are relevant.
		Enumeration<Instance> enu = trainInstances.enumerateInstances();
		i = 0;
		while (enu.hasMoreElements()) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			instance = enu.nextElement();
			if (instance.isMissing(this.m_attIndex)) {
				break;
			}
			this.m_distribution.add(1, instance);
			i++;
		}
		firstMiss = i;

		// Compute minimum number of Instances required in each
		// subset.
		minSplit = 0.1 * (this.m_distribution.total()) / (trainInstances.numClasses());
		if (Utils.smOrEq(minSplit, this.m_minNoObj)) {
			minSplit = this.m_minNoObj;
		} else if (Utils.gr(minSplit, 25)) {
			minSplit = 25;
		}

		// Enough Instances with known values?
		if (Utils.sm(firstMiss, 2 * minSplit)) {
			return;
		}

		// Compute values of criteria for all possible split
		// indices.
		defaultEnt = infoGainCrit.oldEnt(this.m_distribution);
		while (next < firstMiss) {

			if (trainInstances.instance(next - 1).value(this.m_attIndex) + 1e-5 < trainInstances.instance(next).value(this.m_attIndex)) {

				// Move class values for all Instances up to next
				// possible split point.
				this.m_distribution.shiftRange(1, 0, trainInstances, last, next);

				// Check if enough Instances in each subset and compute
				// values for criteria.
				if (Utils.grOrEq(this.m_distribution.perBag(0), minSplit) && Utils.grOrEq(this.m_distribution.perBag(1), minSplit)) {
					currentInfoGain = infoGainCrit.splitCritValue(this.m_distribution, this.m_sumOfWeights, defaultEnt);
					if (Utils.gr(currentInfoGain, this.m_infoGain)) {
						this.m_infoGain = currentInfoGain;
						splitIndex = next - 1;
					}
					this.m_index++;
				}
				last = next;
			}
			next++;
		}

		// Was there any useful split?
		if (this.m_index == 0) {
			return;
		}

		// Compute modified information gain for best split.
		if (this.m_useMDLcorrection) {
			this.m_infoGain = this.m_infoGain - (Utils.log2(this.m_index) / this.m_sumOfWeights);
		}
		if (Utils.smOrEq(this.m_infoGain, 0)) {
			return;
		}

		// Set instance variables' values to values for
		// best split.
		this.m_numSubsets = 2;
		this.m_splitPoint = (trainInstances.instance(splitIndex + 1).value(this.m_attIndex) + trainInstances.instance(splitIndex).value(this.m_attIndex)) / 2;

		// In case we have a numerical precision problem we need to choose the
		// smaller value
		if (this.m_splitPoint == trainInstances.instance(splitIndex + 1).value(this.m_attIndex)) {
			this.m_splitPoint = trainInstances.instance(splitIndex).value(this.m_attIndex);
		}

		// Restore distributioN for best split.
		this.m_distribution = new Distribution(2, trainInstances.numClasses());
		this.m_distribution.addRange(0, trainInstances, 0, splitIndex + 1);
		this.m_distribution.addRange(1, trainInstances, splitIndex + 1, firstMiss);

		// Compute modified gain ratio for best split.
		this.m_gainRatio = gainRatioCrit.splitCritValue(this.m_distribution, this.m_sumOfWeights, this.m_infoGain);
	}

	/**
	 * Returns (C4.5-type) information gain for the generated split.
	 */
	public final double infoGain() {

		return this.m_infoGain;
	}

	/**
	 * Prints left side of condition..
	 *
	 * @param data
	 *          training set.
	 */
	@Override
	public final String leftSide(final Instances data) {

		return data.attribute(this.m_attIndex).name();
	}

	/**
	 * Prints the condition satisfied by instances in a subset.
	 *
	 * @param index
	 *          of subset
	 * @param data
	 *          training set.
	 */
	@Override
	public final String rightSide(final int index, final Instances data) {

		StringBuffer text;

		text = new StringBuffer();
		if (data.attribute(this.m_attIndex).isNominal()) {
			text.append(" = " + data.attribute(this.m_attIndex).value(index));
		} else if (index == 0) {
			text.append(" <= " + Utils.doubleToString(this.m_splitPoint, 6));
		} else {
			text.append(" > " + Utils.doubleToString(this.m_splitPoint, 6));
		}
		return text.toString();
	}

	/**
	 * Returns a string containing java source code equivalent to the test made at this node. The
	 * instance being tested is called "i".
	 *
	 * @param index
	 *          index of the nominal value tested
	 * @param data
	 *          the data containing instance structure info
	 * @return a value of type 'String'
	 */
	@Override
	public final String sourceExpression(final int index, final Instances data) {

		StringBuffer expr = null;
		if (index < 0) {
			return "i[" + this.m_attIndex + "] == null";
		}
		if (data.attribute(this.m_attIndex).isNominal()) {
			expr = new StringBuffer("i[");
			expr.append(this.m_attIndex).append("]");
			expr.append(".equals(\"").append(data.attribute(this.m_attIndex).value(index)).append("\")");
		} else {
			expr = new StringBuffer("((Double) i[");
			expr.append(this.m_attIndex).append("])");
			if (index == 0) {
				expr.append(".doubleValue() <= ").append(this.m_splitPoint);
			} else {
				expr.append(".doubleValue() > ").append(this.m_splitPoint);
			}
		}
		return expr.toString();
	}

	/**
	 * Sets split point to greatest value in given data smaller or equal to old split point. (C4.5 does
	 * this for some strange reason).
	 */
	public final void setSplitPoint(final Instances allInstances) {

		double newSplitPoint = -Double.MAX_VALUE;
		double tempValue;
		Instance instance;

		if ((allInstances.attribute(this.m_attIndex).isNumeric()) && (this.m_numSubsets > 1)) {
			Enumeration<Instance> enu = allInstances.enumerateInstances();
			while (enu.hasMoreElements()) {
				instance = enu.nextElement();
				if (!instance.isMissing(this.m_attIndex)) {
					tempValue = instance.value(this.m_attIndex);
					if (Utils.gr(tempValue, newSplitPoint) && Utils.smOrEq(tempValue, this.m_splitPoint)) {
						newSplitPoint = tempValue;
					}
				}
			}
			this.m_splitPoint = newSplitPoint;
		}
	}

	/**
	 * Returns the minsAndMaxs of the index.th subset.
	 */
	public final double[][] minsAndMaxs(final Instances data, final double[][] minsAndMaxs, final int index) {

		double[][] newMinsAndMaxs = new double[data.numAttributes()][2];

		for (int i = 0; i < data.numAttributes(); i++) {
			newMinsAndMaxs[i][0] = minsAndMaxs[i][0];
			newMinsAndMaxs[i][1] = minsAndMaxs[i][1];
			if (i == this.m_attIndex) {
				if (data.attribute(this.m_attIndex).isNominal()) {
					newMinsAndMaxs[this.m_attIndex][1] = 1;
				} else {
					newMinsAndMaxs[this.m_attIndex][1 - index] = this.m_splitPoint;
				}
			}
		}

		return newMinsAndMaxs;
	}

	/**
	 * Sets distribution associated with model.
	 */
	@Override
	public void resetDistribution(final Instances data) throws Exception {

		Instances insts = new Instances(data, data.numInstances());
		for (int i = 0; i < data.numInstances(); i++) {
			if (this.whichSubset(data.instance(i)) > -1) {
				insts.add(data.instance(i));
			}
		}
		Distribution newD = new Distribution(insts, this);
		newD.addInstWithUnknown(data, this.m_attIndex);
		this.m_distribution = newD;
	}

	/**
	 * Returns weights if instance is assigned to more than one subset. Returns null if instance is only
	 * assigned to one subset.
	 */
	@Override
	public final double[] weights(final Instance instance) {

		double[] weights;
		int i;

		if (instance.isMissing(this.m_attIndex)) {
			weights = new double[this.m_numSubsets];
			for (i = 0; i < this.m_numSubsets; i++) {
				weights[i] = this.m_distribution.perBag(i) / this.m_distribution.total();
			}
			return weights;
		} else {
			return null;
		}
	}

	/**
	 * Returns index of subset instance is assigned to. Returns -1 if instance is assigned to more than
	 * one subset.
	 *
	 * @exception Exception
	 *              if something goes wrong
	 */
	@Override
	public final int whichSubset(final Instance instance) throws Exception {

		if (instance.isMissing(this.m_attIndex)) {
			return -1;
		} else {
			if (instance.attribute(this.m_attIndex).isNominal()) {
				return (int) instance.value(this.m_attIndex);
			} else if (Utils.smOrEq(instance.value(this.m_attIndex), this.m_splitPoint)) {
				return 0;
			} else {
				return 1;
			}
		}
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
