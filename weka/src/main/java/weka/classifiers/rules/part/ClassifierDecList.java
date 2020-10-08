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
 *    ClassifierDecList.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.rules.part;

import java.io.Serializable;

import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.EntropySplitCrit;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.NoSplit;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * Class for handling a rule (partial tree) for a decision list.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class ClassifierDecList implements Serializable, RevisionHandler {

	/** for serialization */
	private static final long serialVersionUID = 7284358349711992497L;

	/** Minimum number of objects */
	protected int m_minNumObj;

	/** To compute the entropy. */
	protected static EntropySplitCrit m_splitCrit = new EntropySplitCrit();

	/** The model selection method. */
	protected ModelSelection m_toSelectModel;

	/** Local model at node. */
	protected ClassifierSplitModel m_localModel;

	/** References to sons. */
	protected ClassifierDecList[] m_sons;

	/** True if node is leaf. */
	protected boolean m_isLeaf;

	/** True if node is empty. */
	protected boolean m_isEmpty;

	/** The training instances. */
	protected Instances m_train;

	/** The pruning instances. */
	protected Distribution m_test;

	/** Which son to expand? */
	protected int indeX;

	/**
	 * Constructor - just calls constructor of class DecList.
	 */
	public ClassifierDecList(final ModelSelection toSelectLocModel, final int minNum) {

		this.m_toSelectModel = toSelectLocModel;
		this.m_minNumObj = minNum;
	}

	/**
	 * Method for building a pruned partial tree.
	 *
	 * @exception Exception
	 *                if something goes wrong
	 */
	public void buildRule(final Instances data) throws Exception {
		// XXX kill weka execution
		if (Thread.interrupted()) {
			throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
		}

		this.buildDecList(data, false);

		this.cleanup(new Instances(data, 0));
	}

	/**
	 * Builds the partial tree without hold out set.
	 *
	 * @exception Exception
	 *                if something goes wrong
	 */
	public void buildDecList(Instances data, final boolean leaf) throws Exception {

		Instances[] localInstances;
		int ind;
		int i, j;
		double sumOfWeights;
		NoSplit noSplit;

		this.m_train = null;
		this.m_test = null;
		this.m_isLeaf = false;
		this.m_isEmpty = false;
		this.m_sons = null;
		this.indeX = 0;
		sumOfWeights = data.sumOfWeights();
		noSplit = new NoSplit(new Distribution(data));
		if (leaf) {
			this.m_localModel = noSplit;
		} else {
			this.m_localModel = this.m_toSelectModel.selectModel(data);
		}
		if (this.m_localModel.numSubsets() > 1) {
			localInstances = this.m_localModel.split(data);
			data = null;
			this.m_sons = new ClassifierDecList[this.m_localModel.numSubsets()];
			i = 0;
			do {
				i++;
				ind = this.chooseIndex();
				if (ind == -1) {
					for (j = 0; j < this.m_sons.length; j++) {
						if (this.m_sons[j] == null) {
							this.m_sons[j] = this.getNewDecList(localInstances[j], true);
						}
					}
					if (i < 2) {
						this.m_localModel = noSplit;
						this.m_isLeaf = true;
						this.m_sons = null;
						if (Utils.eq(sumOfWeights, 0)) {
							this.m_isEmpty = true;
						}
						return;
					}
					ind = 0;
					break;
				} else {
					this.m_sons[ind] = this.getNewDecList(localInstances[ind], false);
				}
			} while ((i < this.m_sons.length) && (this.m_sons[ind].m_isLeaf));

			// Choose rule
			this.indeX = this.chooseLastIndex();
		} else {
			this.m_isLeaf = true;
			if (Utils.eq(sumOfWeights, 0)) {
				this.m_isEmpty = true;
			}
		}
	}

	/**
	 * Classifies an instance.
	 *
	 * @exception Exception
	 *                if something goes wrong
	 */
	public double classifyInstance(final Instance instance) throws Exception {

		double maxProb = -1;
		double currentProb;
		int maxIndex = 0;
		int j;

		for (j = 0; j < instance.numClasses(); j++) {
			currentProb = this.getProbs(j, instance, 1);
			if (Utils.gr(currentProb, maxProb)) {
				maxIndex = j;
				maxProb = currentProb;
			}
		}
		if (Utils.eq(maxProb, 0)) {
			return -1.0;
		} else {
			return maxIndex;
		}
	}

	/**
	 * Returns class probabilities for a weighted instance.
	 *
	 * @exception Exception
	 *                if something goes wrong
	 */
	public final double[] distributionForInstance(final Instance instance) throws Exception {

		double[] doubles = new double[instance.numClasses()];

		for (int i = 0; i < doubles.length; i++) {
			doubles[i] = this.getProbs(i, instance, 1);
		}

		return doubles;
	}

	/**
	 * Returns the weight a rule assigns to an instance.
	 *
	 * @exception Exception
	 *                if something goes wrong
	 */
	public double weight(final Instance instance) throws Exception {

		int subset;

		if (this.m_isLeaf) {
			return 1;
		}
		subset = this.m_localModel.whichSubset(instance);
		if (subset == -1) {
			return (this.m_localModel.weights(instance))[this.indeX] * this.m_sons[this.indeX].weight(instance);
		}
		if (subset == this.indeX) {
			return this.m_sons[this.indeX].weight(instance);
		}
		return 0;
	}

	/**
	 * Cleanup in order to save memory.
	 *
	 * @throws InterruptedException
	 */
	public final void cleanup(final Instances justHeaderInfo) throws InterruptedException {
		this.m_train = justHeaderInfo;
		this.m_test = null;
		if (!this.m_isLeaf) {
			for (ClassifierDecList m_son : this.m_sons) {
				// XXX interrupt weka
				if (Thread.interrupted()) {
					throw new InterruptedException("Killed WEKA!");
				}
				if (m_son != null) {
					m_son.cleanup(justHeaderInfo);
				}
			}
		}
	}

	/**
	 * Prints rules.
	 */
	@Override
	public String toString() {

		try {
			StringBuffer text;

			text = new StringBuffer();
			if (this.m_isLeaf) {
				text.append(": ");
				text.append(this.m_localModel.dumpLabel(0, this.m_train) + "\n");
			} else {
				this.dumpDecList(text);
				// dumpTree(0,text);
			}
			return text.toString();
		} catch (Exception e) {
			return "Can't print rule.";
		}
	}

	/**
	 * Returns a newly created tree.
	 *
	 * @exception Exception
	 *                if something goes wrong
	 */
	protected ClassifierDecList getNewDecList(final Instances train, final boolean leaf) throws Exception {

		ClassifierDecList newDecList = new ClassifierDecList(this.m_toSelectModel, this.m_minNumObj);
		newDecList.buildDecList(train, leaf);

		return newDecList;
	}

	/**
	 * Method for choosing a subset to expand.
	 */
	public final int chooseIndex() {

		int minIndex = -1;
		double estimated, min = Double.MAX_VALUE;
		int i, j;

		for (i = 0; i < this.m_sons.length; i++) {
			if (this.son(i) == null) {
				if (Utils.sm(this.localModel().distribution().perBag(i), this.m_minNumObj)) {
					estimated = Double.MAX_VALUE;
				} else {
					estimated = 0;
					for (j = 0; j < this.localModel().distribution().numClasses(); j++) {
						estimated -= m_splitCrit.lnFunc(this.localModel().distribution().perClassPerBag(i, j));
					}
					estimated += m_splitCrit.lnFunc(this.localModel().distribution().perBag(i));
					estimated /= (this.localModel().distribution().perBag(i) * ContingencyTables.log2);
				}
				if (Utils.smOrEq(estimated, 0)) {
					return i;
				}
				if (Utils.sm(estimated, min)) {
					min = estimated;
					minIndex = i;
				}
			}
		}

		return minIndex;
	}

	/**
	 * Choose last index (ie. choose rule).
	 */
	public final int chooseLastIndex() {

		int minIndex = 0;
		double estimated, min = Double.MAX_VALUE;

		if (!this.m_isLeaf) {
			for (int i = 0; i < this.m_sons.length; i++) {
				if (this.son(i) != null) {
					if (Utils.grOrEq(this.localModel().distribution().perBag(i), this.m_minNumObj)) {
						estimated = this.son(i).getSizeOfBranch();
						if (Utils.sm(estimated, min)) {
							min = estimated;
							minIndex = i;
						}
					}
				}
			}
		}

		return minIndex;
	}

	/**
	 * Returns the number of instances covered by a branch
	 */
	protected double getSizeOfBranch() {

		if (this.m_isLeaf) {
			return -this.localModel().distribution().total();
		} else {
			return this.son(this.indeX).getSizeOfBranch();
		}
	}

	/**
	 * Help method for printing tree structure.
	 */
	private void dumpDecList(final StringBuffer text) throws Exception {

		text.append(this.m_localModel.leftSide(this.m_train));
		text.append(this.m_localModel.rightSide(this.indeX, this.m_train));
		if (this.m_sons[this.indeX].m_isLeaf) {
			text.append(": ");
			text.append(this.m_localModel.dumpLabel(this.indeX, this.m_train) + "\n");
		} else {
			text.append(" AND\n");
			this.m_sons[this.indeX].dumpDecList(text);
		}
	}

	/**
	 * Help method for computing class probabilities of a given instance.
	 *
	 * @exception Exception
	 *                Exception if something goes wrong
	 */
	private double getProbs(final int classIndex, final Instance instance, final double weight) throws Exception {
		if (Thread.interrupted()) {
			throw new InterruptedException("Killed WEKA!");
		}

		double[] weights;
		int treeIndex;

		if (this.m_isLeaf) {
			return weight * this.localModel().classProb(classIndex, instance, -1);
		} else {
			treeIndex = this.localModel().whichSubset(instance);
			if (treeIndex == -1) {
				weights = this.localModel().weights(instance);
				return this.son(this.indeX).getProbs(classIndex, instance, weights[this.indeX] * weight);
			} else {
				if (treeIndex == this.indeX) {
					return this.son(this.indeX).getProbs(classIndex, instance, weight);
				} else {
					return 0;
				}
			}
		}
	}

	/**
	 * Method just exists to make program easier to read.
	 */
	protected ClassifierSplitModel localModel() {

		return this.m_localModel;
	}

	/**
	 * Method just exists to make program easier to read.
	 */
	protected ClassifierDecList son(final int index) {

		return this.m_sons[index];
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
