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
 *    C45PruneableDecList.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.rules.part;

import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.NoSplit;
import weka.classifiers.trees.j48.Stats;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * Class for handling a partial tree structure pruned using C4.5's pruning heuristic.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class C45PruneableDecList extends ClassifierDecList {

	/** for serialization */
	private static final long serialVersionUID = -2757684345218324559L;

	/** CF */
	private double CF = 0.25;

	/**
	 * Constructor for pruneable tree structure. Stores reference to associated training data at each node.
	 *
	 * @param toSelectLocModel
	 *            selection method for local splitting model
	 * @param cf
	 *            the confidence factor for pruning
	 * @param minNum
	 *            the minimum number of objects in a leaf
	 * @exception Exception
	 *                if something goes wrong
	 */
	public C45PruneableDecList(final ModelSelection toSelectLocModel, final double cf, final int minNum) throws Exception {

		super(toSelectLocModel, minNum);

		this.CF = cf;
	}

	/**
	 * Builds the partial tree without hold out set.
	 *
	 * @exception Exception
	 *                if something goes wrong
	 */
	@Override
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
				// XXX kill weka execution
				if (Thread.currentThread().isInterrupted()) {
					throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
				}
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
	 * @exception Exception
	 *                if something goes wrong
	 */
	@Override
	protected ClassifierDecList getNewDecList(final Instances data, final boolean leaf) throws Exception {

		C45PruneableDecList newDecList = new C45PruneableDecList(this.m_toSelectModel, this.CF, this.m_minNumObj);

		newDecList.buildDecList(data, leaf);

		return newDecList;
	}

	/**
	 * Prunes the end of the rule.
	 */
	protected void pruneEnd() {

		double errorsLeaf, errorsTree;

		errorsTree = this.getEstimatedErrorsForTree();
		errorsLeaf = this.getEstimatedErrorsForLeaf();
		if (Utils.smOrEq(errorsLeaf, errorsTree + 0.1)) { // +0.1 as in C4.5
			this.m_isLeaf = true;
			this.m_sons = null;
			this.m_localModel = new NoSplit(this.localModel().distribution());
		}
	}

	/**
	 * Computes estimated errors for tree.
	 */
	private double getEstimatedErrorsForTree() {

		if (this.m_isLeaf) {
			return this.getEstimatedErrorsForLeaf();
		} else {
			double error = 0;
			for (int i = 0; i < this.m_sons.length; i++) {
				if (!Utils.eq(this.son(i).localModel().distribution().total(), 0)) {
					error += ((C45PruneableDecList) this.son(i)).getEstimatedErrorsForTree();
				}
			}
			return error;
		}
	}

	/**
	 * Computes estimated errors for leaf.
	 */
	public double getEstimatedErrorsForLeaf() {

		double errors = this.localModel().distribution().numIncorrect();

		return errors + Stats.addErrs(this.localModel().distribution().total(), errors, (float) this.CF);
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
