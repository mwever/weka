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
 * HillClimber.java
 * Copyright (C) 2004-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.bayes.net.search.local;

import java.io.Serializable;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.ParentSet;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * <!-- globalinfo-start --> This Bayes Network learning algorithm uses a hill climbing algorithm adding, deleting and reversing arcs. The search is not restricted by an order on the variables (unlike K2). The difference with B and B2 is
 * that this hill climber also considers arrows part of the naive Bayes structure for deletion.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -P &lt;nr of parents&gt;
 *  Maximum number of parents
 * </pre>
 *
 * <pre>
 * -R
 *  Use arc reversal operation.
 *  (default false)
 * </pre>
 *
 * <pre>
 * -N
 *  Initial structure is empty (instead of Naive Bayes)
 * </pre>
 *
 * <pre>
 * -mbc
 *  Applies a Markov Blanket correction to the network structure,
 *  after a network structure is learned. This ensures that all
 *  nodes in the network are part of the Markov blanket of the
 *  classifier node.
 * </pre>
 *
 * <pre>
 * -S [BAYES|MDL|ENTROPY|AIC|CROSS_CLASSIC|CROSS_BAYES]
 *  Score type (BAYES, BDeu, MDL, ENTROPY and AIC)
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Remco Bouckaert (rrb@xm.co.nz)
 * @version $Revision$
 */
public class HillClimber extends LocalScoreSearchAlgorithm {

	/** for serialization */
	static final long serialVersionUID = 4322783593818122403L;

	/**
	 * the Operation class contains info on operations performed on the current Bayesian network.
	 */
	class Operation implements Serializable, RevisionHandler {

		/** for serialization */
		static final long serialVersionUID = -4880888790432547895L;

		// constants indicating the type of an operation
		final static int OPERATION_ADD = 0;
		final static int OPERATION_DEL = 1;
		final static int OPERATION_REVERSE = 2;

		/**
		 * c'tor
		 */
		public Operation() {
		}

		/**
		 * c'tor + initializers
		 * 
		 * @param nTail
		 * @param nHead
		 * @param nOperation
		 */
		public Operation(final int nTail, final int nHead, final int nOperation) {
			this.m_nHead = nHead;
			this.m_nTail = nTail;
			this.m_nOperation = nOperation;
		}

		/**
		 * compare this operation with another
		 * 
		 * @param other
		 *            operation to compare with
		 * @return true if operation is the same
		 */
		public boolean equals(final Operation other) {
			if (other == null) {
				return false;
			}
			return ((this.m_nOperation == other.m_nOperation) && (this.m_nHead == other.m_nHead) && (this.m_nTail == other.m_nTail));
		} // equals

		/** number of the tail node **/
		public int m_nTail;

		/** number of the head node **/
		public int m_nHead;

		/** type of operation (ADD, DEL, REVERSE) **/
		public int m_nOperation;

		/** change of score due to this operation **/
		public double m_fDeltaScore = -1E100;

		/**
		 * Returns the revision string.
		 * 
		 * @return the revision
		 */
		@Override
		public String getRevision() {
			return RevisionUtils.extract("$Revision$");
		}
	} // class Operation

	/**
	 * cache for remembering the change in score for steps in the search space
	 */
	class Cache implements RevisionHandler {

		/** change in score due to adding an arc **/
		double[][] m_fDeltaScoreAdd;
		/** change in score due to deleting an arc **/
		double[][] m_fDeltaScoreDel;

		/**
		 * c'tor
		 * 
		 * @param nNrOfNodes
		 *            number of nodes in network, used to determine memory size to reserve
		 */
		Cache(final int nNrOfNodes) {
			this.m_fDeltaScoreAdd = new double[nNrOfNodes][nNrOfNodes];
			this.m_fDeltaScoreDel = new double[nNrOfNodes][nNrOfNodes];
		}

		/**
		 * set cache entry
		 * 
		 * @param oOperation
		 *            operation to perform
		 * @param fValue
		 *            value to put in cache
		 */
		public void put(final Operation oOperation, final double fValue) {
			if (oOperation.m_nOperation == Operation.OPERATION_ADD) {
				this.m_fDeltaScoreAdd[oOperation.m_nTail][oOperation.m_nHead] = fValue;
			} else {
				this.m_fDeltaScoreDel[oOperation.m_nTail][oOperation.m_nHead] = fValue;
			}
		} // put

		/**
		 * get cache entry
		 * 
		 * @param oOperation
		 *            operation to perform
		 * @return cache value
		 */
		public double get(final Operation oOperation) {
			switch (oOperation.m_nOperation) {
			case Operation.OPERATION_ADD:
				return this.m_fDeltaScoreAdd[oOperation.m_nTail][oOperation.m_nHead];
			case Operation.OPERATION_DEL:
				return this.m_fDeltaScoreDel[oOperation.m_nTail][oOperation.m_nHead];
			case Operation.OPERATION_REVERSE:
				return this.m_fDeltaScoreDel[oOperation.m_nTail][oOperation.m_nHead] + this.m_fDeltaScoreAdd[oOperation.m_nHead][oOperation.m_nTail];
			}
			// should never get here
			return 0;
		} // get

		/**
		 * Returns the revision string.
		 * 
		 * @return the revision
		 */
		@Override
		public String getRevision() {
			return RevisionUtils.extract("$Revision$");
		}
	} // class Cache

	/** cache for storing score differences **/
	Cache m_Cache = null;

	/** use the arc reversal operator **/
	boolean m_bUseArcReversal = false;

	/**
	 * search determines the network structure/graph of the network with the Taby algorithm.
	 * 
	 * @param bayesNet
	 *            the network to use
	 * @param instances
	 *            the data to use
	 * @throws Exception
	 *             if something goes wrong
	 */
	@Override
	protected void search(final BayesNet bayesNet, final Instances instances) throws Exception {
		this.initCache(bayesNet, instances);

		// go do the search
		Operation oOperation = this.getOptimalOperation(bayesNet, instances);
		while ((oOperation != null) && (oOperation.m_fDeltaScore > 0)) {
			// XXX Interrupt weka
			if (Thread.currentThread().isInterrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			this.performOperation(bayesNet, instances, oOperation);
			oOperation = this.getOptimalOperation(bayesNet, instances);
		}

		// free up memory
		this.m_Cache = null;
	} // search

	/**
	 * initCache initializes the cache
	 * 
	 * @param bayesNet
	 *            Bayes network to be learned
	 * @param instances
	 *            data set to learn from
	 * @throws Exception
	 *             if something goes wrong
	 */
	void initCache(final BayesNet bayesNet, final Instances instances) throws Exception {

		// determine base scores
		double[] fBaseScores = new double[instances.numAttributes()];
		int nNrOfAtts = instances.numAttributes();

		this.m_Cache = new Cache(nNrOfAtts);

		for (int iAttribute = 0; iAttribute < nNrOfAtts; iAttribute++) {
			this.updateCache(iAttribute, nNrOfAtts, bayesNet.getParentSet(iAttribute));
		}

		for (int iAttribute = 0; iAttribute < nNrOfAtts; iAttribute++) {
			fBaseScores[iAttribute] = this.calcNodeScore(iAttribute);
		}

		for (int iAttributeHead = 0; iAttributeHead < nNrOfAtts; iAttributeHead++) {
			for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
				if (iAttributeHead != iAttributeTail) {
					Operation oOperation = new Operation(iAttributeTail, iAttributeHead, Operation.OPERATION_ADD);
					this.m_Cache.put(oOperation, this.calcScoreWithExtraParent(iAttributeHead, iAttributeTail) - fBaseScores[iAttributeHead]);
				}
			}
		}

	} // initCache

	/**
	 * check whether the operation is not in the forbidden. For base hill climber, there are no restrictions on operations, so we always return true.
	 * 
	 * @param oOperation
	 *            operation to be checked
	 * @return true if operation is not in the tabu list
	 */
	boolean isNotTabu(final Operation oOperation) {
		return true;
	} // isNotTabu

	/**
	 * getOptimalOperation finds the optimal operation that can be performed on the Bayes network that is not in the tabu list.
	 * 
	 * @param bayesNet
	 *            Bayes network to apply operation on
	 * @param instances
	 *            data set to learn from
	 * @return optimal operation found
	 * @throws Exception
	 *             if something goes wrong
	 */
	Operation getOptimalOperation(final BayesNet bayesNet, final Instances instances) throws Exception {
		Operation oBestOperation = new Operation();

		// Add???
		oBestOperation = this.findBestArcToAdd(bayesNet, instances, oBestOperation);
		// Delete???
		oBestOperation = this.findBestArcToDelete(bayesNet, instances, oBestOperation);
		// Reverse???
		if (this.getUseArcReversal()) {
			oBestOperation = this.findBestArcToReverse(bayesNet, instances, oBestOperation);
		}

		// did we find something?
		if (oBestOperation.m_fDeltaScore == -1E100) {
			return null;
		}

		return oBestOperation;
	} // getOptimalOperation

	/**
	 * performOperation applies an operation on the Bayes network and update the cache.
	 * 
	 * @param bayesNet
	 *            Bayes network to apply operation on
	 * @param instances
	 *            data set to learn from
	 * @param oOperation
	 *            operation to perform
	 * @throws Exception
	 *             if something goes wrong
	 */
	void performOperation(final BayesNet bayesNet, final Instances instances, final Operation oOperation) throws Exception {
		// perform operation
		switch (oOperation.m_nOperation) {
		case Operation.OPERATION_ADD:
			this.applyArcAddition(bayesNet, oOperation.m_nHead, oOperation.m_nTail, instances);
			if (bayesNet.getDebug()) {
				System.out.print("Add " + oOperation.m_nHead + " -> " + oOperation.m_nTail);
			}
			break;
		case Operation.OPERATION_DEL:
			this.applyArcDeletion(bayesNet, oOperation.m_nHead, oOperation.m_nTail, instances);
			if (bayesNet.getDebug()) {
				System.out.print("Del " + oOperation.m_nHead + " -> " + oOperation.m_nTail);
			}
			break;
		case Operation.OPERATION_REVERSE:
			this.applyArcDeletion(bayesNet, oOperation.m_nHead, oOperation.m_nTail, instances);
			this.applyArcAddition(bayesNet, oOperation.m_nTail, oOperation.m_nHead, instances);
			if (bayesNet.getDebug()) {
				System.out.print("Rev " + oOperation.m_nHead + " -> " + oOperation.m_nTail);
			}
			break;
		}
	} // performOperation

	/**
	 * 
	 * @param bayesNet
	 * @param iHead
	 * @param iTail
	 * @param instances
	 */
	void applyArcAddition(final BayesNet bayesNet, final int iHead, final int iTail, final Instances instances) {
		ParentSet bestParentSet = bayesNet.getParentSet(iHead);
		bestParentSet.addParent(iTail, instances);
		this.updateCache(iHead, instances.numAttributes(), bestParentSet);
	} // applyArcAddition

	/**
	 * 
	 * @param bayesNet
	 * @param iHead
	 * @param iTail
	 * @param instances
	 */
	void applyArcDeletion(final BayesNet bayesNet, final int iHead, final int iTail, final Instances instances) {
		ParentSet bestParentSet = bayesNet.getParentSet(iHead);
		bestParentSet.deleteParent(iTail, instances);
		this.updateCache(iHead, instances.numAttributes(), bestParentSet);
	} // applyArcAddition

	/**
	 * find best (or least bad) arc addition operation
	 * 
	 * @param bayesNet
	 *            Bayes network to add arc to
	 * @param instances
	 *            data set
	 * @param oBestOperation
	 * @return Operation containing best arc to add, or null if no arc addition is allowed (this can happen if any arc addition introduces a cycle, or all parent sets are filled up to the maximum nr of parents).
	 */
	Operation findBestArcToAdd(final BayesNet bayesNet, final Instances instances, Operation oBestOperation) {
		int nNrOfAtts = instances.numAttributes();
		// find best arc to add
		for (int iAttributeHead = 0; iAttributeHead < nNrOfAtts; iAttributeHead++) {
			if (bayesNet.getParentSet(iAttributeHead).getNrOfParents() < this.m_nMaxNrOfParents) {
				for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
					if (this.addArcMakesSense(bayesNet, instances, iAttributeHead, iAttributeTail)) {
						Operation oOperation = new Operation(iAttributeTail, iAttributeHead, Operation.OPERATION_ADD);
						if (this.m_Cache.get(oOperation) > oBestOperation.m_fDeltaScore) {
							if (this.isNotTabu(oOperation)) {
								oBestOperation = oOperation;
								oBestOperation.m_fDeltaScore = this.m_Cache.get(oOperation);
							}
						}
					}
				}
			}
		}
		return oBestOperation;
	} // findBestArcToAdd

	/**
	 * find best (or least bad) arc deletion operation
	 * 
	 * @param bayesNet
	 *            Bayes network to delete arc from
	 * @param instances
	 *            data set
	 * @param oBestOperation
	 * @return Operation containing best arc to delete, or null if no deletion can be made (happens when there is no arc in the network yet).
	 */
	Operation findBestArcToDelete(final BayesNet bayesNet, final Instances instances, Operation oBestOperation) {
		int nNrOfAtts = instances.numAttributes();
		// find best arc to delete
		for (int iNode = 0; iNode < nNrOfAtts; iNode++) {
			ParentSet parentSet = bayesNet.getParentSet(iNode);
			for (int iParent = 0; iParent < parentSet.getNrOfParents(); iParent++) {
				Operation oOperation = new Operation(parentSet.getParent(iParent), iNode, Operation.OPERATION_DEL);
				if (this.m_Cache.get(oOperation) > oBestOperation.m_fDeltaScore) {
					if (this.isNotTabu(oOperation)) {
						oBestOperation = oOperation;
						oBestOperation.m_fDeltaScore = this.m_Cache.get(oOperation);
					}
				}
			}
		}
		return oBestOperation;
	} // findBestArcToDelete

	/**
	 * find best (or least bad) arc reversal operation
	 * 
	 * @param bayesNet
	 *            Bayes network to reverse arc in
	 * @param instances
	 *            data set
	 * @param oBestOperation
	 * @return Operation containing best arc to reverse, or null if no reversal is allowed (happens if there is no arc in the network yet, or when any such reversal introduces a cycle).
	 */
	Operation findBestArcToReverse(final BayesNet bayesNet, final Instances instances, Operation oBestOperation) {
		int nNrOfAtts = instances.numAttributes();
		// find best arc to reverse
		for (int iNode = 0; iNode < nNrOfAtts; iNode++) {
			ParentSet parentSet = bayesNet.getParentSet(iNode);
			for (int iParent = 0; iParent < parentSet.getNrOfParents(); iParent++) {
				int iTail = parentSet.getParent(iParent);
				// is reversal allowed?
				if (this.reverseArcMakesSense(bayesNet, instances, iNode, iTail) && bayesNet.getParentSet(iTail).getNrOfParents() < this.m_nMaxNrOfParents) {
					// go check if reversal results in the best step forward
					Operation oOperation = new Operation(parentSet.getParent(iParent), iNode, Operation.OPERATION_REVERSE);
					if (this.m_Cache.get(oOperation) > oBestOperation.m_fDeltaScore) {
						if (this.isNotTabu(oOperation)) {
							oBestOperation = oOperation;
							oBestOperation.m_fDeltaScore = this.m_Cache.get(oOperation);
						}
					}
				}
			}
		}
		return oBestOperation;
	} // findBestArcToReverse

	/**
	 * update the cache due to change of parent set of a node
	 * 
	 * @param iAttributeHead
	 *            node that has its parent set changed
	 * @param nNrOfAtts
	 *            number of nodes/attributes in data set
	 * @param parentSet
	 *            new parents set of node iAttributeHead
	 */
	void updateCache(final int iAttributeHead, final int nNrOfAtts, final ParentSet parentSet) {
		// update cache entries for arrows heading towards iAttributeHead
		double fBaseScore = this.calcNodeScore(iAttributeHead);
		int nNrOfParents = parentSet.getNrOfParents();
		for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
			if (iAttributeTail != iAttributeHead) {
				if (!parentSet.contains(iAttributeTail)) {
					// add entries to cache for adding arcs
					if (nNrOfParents < this.m_nMaxNrOfParents) {
						Operation oOperation = new Operation(iAttributeTail, iAttributeHead, Operation.OPERATION_ADD);
						this.m_Cache.put(oOperation, this.calcScoreWithExtraParent(iAttributeHead, iAttributeTail) - fBaseScore);
					}
				} else {
					// add entries to cache for deleting arcs
					Operation oOperation = new Operation(iAttributeTail, iAttributeHead, Operation.OPERATION_DEL);
					this.m_Cache.put(oOperation, this.calcScoreWithMissingParent(iAttributeHead, iAttributeTail) - fBaseScore);
				}
			}
		}
	} // updateCache

	/**
	 * Sets the max number of parents
	 * 
	 * @param nMaxNrOfParents
	 *            the max number of parents
	 */
	public void setMaxNrOfParents(final int nMaxNrOfParents) {
		this.m_nMaxNrOfParents = nMaxNrOfParents;
	}

	/**
	 * Gets the max number of parents.
	 * 
	 * @return the max number of parents
	 */
	public int getMaxNrOfParents() {
		return this.m_nMaxNrOfParents;
	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(4);

		newVector.addElement(new Option("\tMaximum number of parents", "P", 1, "-P <nr of parents>"));
		newVector.addElement(new Option("\tUse arc reversal operation.\n\t(default false)", "R", 0, "-R"));
		newVector.addElement(new Option("\tInitial structure is empty (instead of Naive Bayes)", "N", 0, "-N"));
		newVector.addElement(new Option("\tInitial structure specified in XML BIF file", "X", 1, "-X"));

		newVector.addAll(Collections.list(super.listOptions()));

		return newVector.elements();
	} // listOptions

	/**
	 * Parses a given list of options.
	 * <p/>
	 * 
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 * 
	 * <pre>
	 * -P &lt;nr of parents&gt;
	 *  Maximum number of parents
	 * </pre>
	 * 
	 * <pre>
	 * -R
	 *  Use arc reversal operation.
	 *  (default false)
	 * </pre>
	 * 
	 * <pre>
	 * -N
	 *  Initial structure is empty (instead of Naive Bayes)
	 * </pre>
	 * 
	 * <pre>
	 * -mbc
	 *  Applies a Markov Blanket correction to the network structure, 
	 *  after a network structure is learned. This ensures that all 
	 *  nodes in the network are part of the Markov blanket of the 
	 *  classifier node.
	 * </pre>
	 * 
	 * <pre>
	 * -S [BAYES|MDL|ENTROPY|AIC|CROSS_CLASSIC|CROSS_BAYES]
	 *  Score type (BAYES, BDeu, MDL, ENTROPY and AIC)
	 * </pre>
	 * 
	 * <!-- options-end -->
	 * 
	 * @param options
	 *            the list of options as an array of strings
	 * @throws Exception
	 *             if an option is not supported
	 */
	@Override
	public void setOptions(final String[] options) throws Exception {
		this.setUseArcReversal(Utils.getFlag('R', options));

		this.setInitAsNaiveBayes(!(Utils.getFlag('N', options)));

		this.m_sInitalBIFFile = Utils.getOption('X', options);

		String sMaxNrOfParents = Utils.getOption('P', options);
		if (sMaxNrOfParents.length() != 0) {
			this.setMaxNrOfParents(Integer.parseInt(sMaxNrOfParents));
		} else {
			this.setMaxNrOfParents(100000);
		}

		super.setOptions(options);
	} // setOptions

	/**
	 * Gets the current settings of the search algorithm.
	 * 
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String[] getOptions() {

		Vector<String> options = new Vector<String>();

		if (this.getUseArcReversal()) {
			options.add("-R");
		}

		if (!this.getInitAsNaiveBayes()) {
			options.add("-N");
		}
		if (this.m_sInitalBIFFile != null && !this.m_sInitalBIFFile.equals("")) {
			options.add("-X");
			options.add(this.m_sInitalBIFFile);
		}

		options.add("-P");
		options.add("" + this.m_nMaxNrOfParents);

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[0]);
	} // getOptions

	/**
	 * Sets whether to init as naive bayes
	 * 
	 * @param bInitAsNaiveBayes
	 *            whether to init as naive bayes
	 */
	public void setInitAsNaiveBayes(final boolean bInitAsNaiveBayes) {
		this.m_bInitAsNaiveBayes = bInitAsNaiveBayes;
	}

	/**
	 * Gets whether to init as naive bayes
	 * 
	 * @return whether to init as naive bayes
	 */
	public boolean getInitAsNaiveBayes() {
		return this.m_bInitAsNaiveBayes;
	}

	/**
	 * get use the arc reversal operation
	 * 
	 * @return whether the arc reversal operation should be used
	 */
	public boolean getUseArcReversal() {
		return this.m_bUseArcReversal;
	} // getUseArcReversal

	/**
	 * set use the arc reversal operation
	 * 
	 * @param bUseArcReversal
	 *            whether the arc reversal operation should be used
	 */
	public void setUseArcReversal(final boolean bUseArcReversal) {
		this.m_bUseArcReversal = bUseArcReversal;
	} // setUseArcReversal

	/**
	 * This will return a string describing the search algorithm.
	 * 
	 * @return The string.
	 */
	@Override
	public String globalInfo() {
		return "This Bayes Network learning algorithm uses a hill climbing algorithm " + "adding, deleting and reversing arcs. The search is not restricted by an order "
				+ "on the variables (unlike K2). The difference with B and B2 is that this hill " + "climber also considers arrows part of the naive Bayes structure for deletion.";
	} // globalInfo

	/**
	 * @return a string to describe the Use Arc Reversal option.
	 */
	public String useArcReversalTipText() {
		return "When set to true, the arc reversal operation is used in the search.";
	} // useArcReversalTipText

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision$");
	}

} // HillClimber
