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
 *    PruneableDecList.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.rules.part;

import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.NoSplit;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * Class for handling a partial tree structure that can be pruned using a pruning set.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class PruneableDecList extends ClassifierDecList {

	/** for serialization */
	private static final long serialVersionUID = -7228103346297172921L;

	/**
	 * Constructor for pruneable partial tree structure.
	 *
	 * @param toSelectLocModel
	 *            selection method for local splitting model
	 * @param minNum
	 *            minimum number of objects in leaf
	 */
	public PruneableDecList(final ModelSelection toSelectLocModel, final int minNum) {
		super(toSelectLocModel, minNum);
	}

	/**
	 * Method for building a pruned partial tree.
	 *
	 * @throws Exception
	 *             if tree can't be built successfully
	 */
	public void buildRule(final Instances train, final Instances test) throws Exception {
		this.buildDecList(train, test, false);
		this.cleanup(new Instances(train, 0));
	}

	/**
	 * Builds the partial tree with hold out set
	 *
	 * @throws Exception
	 *             if something goes wrong
	 */
	public void buildDecList(Instances train, Instances test, final boolean leaf) throws Exception {

		Instances[] localTrain, localTest;
		int ind;
		int i, j;
		double sumOfWeights;
		NoSplit noSplit;

		this.m_train = null;
		this.m_isLeaf = false;
		this.m_isEmpty = false;
		this.m_sons = null;
		this.indeX = 0;
		sumOfWeights = train.sumOfWeights();
		noSplit = new NoSplit(new Distribution(train));
		if (leaf) {
			this.m_localModel = noSplit;
		} else {
			this.m_localModel = this.m_toSelectModel.selectModel(train, test);
		}
		this.m_test = new Distribution(test, this.m_localModel);
		if (this.m_localModel.numSubsets() > 1) {
			localTrain = this.m_localModel.split(train);
			localTest = this.m_localModel.split(test);
			train = null;
			test = null;
			this.m_sons = new ClassifierDecList[this.m_localModel.numSubsets()];
			i = 0;
			do {
				// XXX interrupt weka
				if (Thread.interrupted()) {
					throw new InterruptedException("Killed WEKA!");
				}
				i++;
				ind = this.chooseIndex();
				if (ind == -1) {
					for (j = 0; j < this.m_sons.length; j++) {
						if (this.m_sons[j] == null) {
							this.m_sons[j] = this.getNewDecList(localTrain[j], localTest[j], true);
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
					this.m_sons[ind] = this.getNewDecList(localTrain[ind], localTest[ind], false);
				}
			} while ((i < this.m_sons.length) && (this.m_sons[ind].m_isLeaf));

			// Check if all successors are leaves
			for (j = 0; j < this.m_sons.length; j++) {
				if ((this.m_sons[j] == null) || (!this.m_sons[j].m_isLeaf)) {
					break;
				}
			}
			if (j == this.m_sons.length) {
				this.pruneEnd();
				if (!this.m_isLeaf) {
					this.indeX = this.chooseLastIndex();
				}
			} else {
				this.indeX = this.chooseLastIndex();
			}
		} else {
			this.m_isLeaf = true;
			if (Utils.eq(sumOfWeights, 0)) {
				this.m_isEmpty = true;
			}
		}
	}

	/**
	 * Returns a newly created tree.
	 *
	 * @param train
	 *            train data
	 * @param test
	 *            test data
	 * @param leaf
	 * @throws Exception
	 *             if something goes wrong
	 */
	protected ClassifierDecList getNewDecList(final Instances train, final Instances test, final boolean leaf) throws Exception {

		PruneableDecList newDecList = new PruneableDecList(this.m_toSelectModel, this.m_minNumObj);

		newDecList.buildDecList(train, test, leaf);

		return newDecList;
	}

	/**
	 * Prunes the end of the rule.
	 */
	protected void pruneEnd() throws Exception {

		double errorsLeaf, errorsTree;

		errorsTree = this.errorsForTree();
		errorsLeaf = this.errorsForLeaf();
		if (Utils.smOrEq(errorsLeaf, errorsTree)) {
			this.m_isLeaf = true;
			this.m_sons = null;
			this.m_localModel = new NoSplit(this.localModel().distribution());
		}
	}

	/**
	 * Computes error estimate for tree.
	 */
	private double errorsForTree() throws Exception {
		if (this.m_isLeaf) {
			return this.errorsForLeaf();
		} else {
			double error = 0;
			for (int i = 0; i < this.m_sons.length; i++) {
				// XXX interrupt weka
				if (Thread.interrupted()) {
					throw new InterruptedException("Killed WEKA!");
				}
				if (Utils.eq(this.son(i).localModel().distribution().total(), 0)) {
					error += this.m_test.perBag(i) - this.m_test.perClassPerBag(i, this.localModel().distribution().maxClass());
				} else {
					error += ((PruneableDecList) this.son(i)).errorsForTree();
				}
			}

			return error;
		}
	}

	/**
	 * Computes estimated errors for leaf.
	 */
	private double errorsForLeaf() throws Exception {
		return this.m_test.total() - this.m_test.perClass(this.localModel().distribution().maxClass());
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
