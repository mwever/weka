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
 *    MarginCalculator.java
 *    Copyright (C) 2007-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.bayes.net;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.Vector;

import weka.classifiers.bayes.BayesNet;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;

public class MarginCalculator implements Serializable, RevisionHandler {
	/** for serialization */
	private static final long serialVersionUID = 650278019241175534L;

	boolean m_debug = false;
	public JunctionTreeNode m_root = null;
	JunctionTreeNode[] jtNodes;

	public int getNode(final String sNodeName) {
		int iNode = 0;
		while (iNode < this.m_root.m_bayesNet.m_Instances.numAttributes()) {
			if (this.m_root.m_bayesNet.m_Instances.attribute(iNode).name().equals(sNodeName)) {
				return iNode;
			}
			iNode++;
		}
		// throw new Exception("Could not find node [[" + sNodeName + "]]");
		return -1;
	}

	public String toXMLBIF03() {
		return this.m_root.m_bayesNet.toXMLBIF03();
	}

	/**
	 * Calc marginal distributions of nodes in Bayesian network Note that a connected network is assumed. Unconnected networks may give unexpected results.
	 *
	 * @param bayesNet
	 */
	public void calcMargins(final BayesNet bayesNet) throws Exception {
		// System.out.println(bayesNet.toString());
		boolean[][] bAdjacencyMatrix = this.moralize(bayesNet);
		this.process(bAdjacencyMatrix, bayesNet);
	} // calcMargins

	public void calcFullMargins(final BayesNet bayesNet) throws Exception {
		// System.out.println(bayesNet.toString());
		int nNodes = bayesNet.getNrOfNodes();
		boolean[][] bAdjacencyMatrix = new boolean[nNodes][nNodes];
		for (int iNode = 0; iNode < nNodes; iNode++) {
			for (int iNode2 = 0; iNode2 < nNodes; iNode2++) {
				bAdjacencyMatrix[iNode][iNode2] = true;
			}
		}
		this.process(bAdjacencyMatrix, bayesNet);
	} // calcMargins

	public void process(boolean[][] bAdjacencyMatrix, final BayesNet bayesNet) throws Exception {
		int[] order = this.getMaxCardOrder(bAdjacencyMatrix);
		bAdjacencyMatrix = this.fillIn(order, bAdjacencyMatrix);
		order = this.getMaxCardOrder(bAdjacencyMatrix);
		Set<Integer>[] cliques = this.getCliques(order, bAdjacencyMatrix);
		Set<Integer>[] separators = this.getSeparators(order, cliques);
		int[] parentCliques = this.getCliqueTree(order, cliques, separators);
		// report cliques
		int nNodes = bAdjacencyMatrix.length;
		if (this.m_debug) {
			for (int i = 0; i < nNodes; i++) {
				int iNode = order[i];
				if (cliques[iNode] != null) {
					Iterator<Integer> nodes = cliques[iNode].iterator();
					while (nodes.hasNext()) {
						int iNode2 = nodes.next();
						System.out.print(iNode2 + " " + bayesNet.getNodeName(iNode2));
						if (nodes.hasNext()) {
							System.out.print(",");
						}
					}
					System.out.print(") S(");
					nodes = separators[iNode].iterator();
					while (nodes.hasNext()) {
						int iNode2 = nodes.next();
						System.out.print(iNode2 + " " + bayesNet.getNodeName(iNode2));
						if (nodes.hasNext()) {
							System.out.print(",");
						}
					}
					System.out.println(") parent clique " + parentCliques[iNode]);
				}
			}
		}

		this.jtNodes = this.getJunctionTree(cliques, separators, parentCliques, order, bayesNet);
		this.m_root = null;
		for (int iNode = 0; iNode < nNodes; iNode++) {
			if (parentCliques[iNode] < 0 && this.jtNodes[iNode] != null) {
				this.m_root = this.jtNodes[iNode];
				break;
			}
		}
		this.m_Margins = new double[nNodes][];
		this.initialize(this.jtNodes, order, cliques, separators, parentCliques);

		// sanity check
		for (int i = 0; i < nNodes; i++) {
			int iNode = order[i];
			if (cliques[iNode] != null) {
				if (parentCliques[iNode] == -1 && separators[iNode].size() > 0) {
					throw new Exception("Something wrong in clique tree");
				}
			}
		}
		if (this.m_debug) {
			// System.out.println(m_root.toString());
		}
	} // process

	void initialize(final JunctionTreeNode[] jtNodes, final int[] order, final Set<Integer>[] cliques, final Set<Integer>[] separators, final int[] parentCliques) {
		int nNodes = order.length;
		for (int i = nNodes - 1; i >= 0; i--) {
			int iNode = order[i];
			if (jtNodes[iNode] != null) {
				jtNodes[iNode].initializeUp();
			}
		}
		for (int i = 0; i < nNodes; i++) {
			int iNode = order[i];
			if (jtNodes[iNode] != null) {
				jtNodes[iNode].initializeDown(false);
			}
		}
	} // initialize

	JunctionTreeNode[] getJunctionTree(final Set<Integer>[] cliques, final Set<Integer>[] separators, final int[] parentCliques, final int[] order, final BayesNet bayesNet) {
		int nNodes = order.length;
		JunctionTreeNode[] jtns = new JunctionTreeNode[nNodes];
		boolean[] bDone = new boolean[nNodes];
		// create junction tree nodes
		for (int i = 0; i < nNodes; i++) {
			int iNode = order[i];
			if (cliques[iNode] != null) {
				jtns[iNode] = new JunctionTreeNode(cliques[iNode], bayesNet, bDone);
			}
		}
		// create junction tree separators
		for (int i = 0; i < nNodes; i++) {
			int iNode = order[i];
			if (cliques[iNode] != null) {
				JunctionTreeNode parent = null;
				if (parentCliques[iNode] > 0) {
					parent = jtns[parentCliques[iNode]];
					JunctionTreeSeparator jts = new JunctionTreeSeparator(separators[iNode], bayesNet, jtns[iNode], parent);
					jtns[iNode].setParentSeparator(jts);
					jtns[parentCliques[iNode]].addChildClique(jtns[iNode]);
				} else {
				}
			}
		}
		return jtns;
	} // getJunctionTree

	public class JunctionTreeSeparator implements Serializable, RevisionHandler {

		private static final long serialVersionUID = 6502780192411755343L;
		int[] m_nNodes;
		int m_nCardinality;
		double[] m_fiParent;
		double[] m_fiChild;
		JunctionTreeNode m_parentNode;
		JunctionTreeNode m_childNode;
		BayesNet m_bayesNet;

		JunctionTreeSeparator(final Set<Integer> separator, final BayesNet bayesNet, final JunctionTreeNode childNode, final JunctionTreeNode parentNode) {
			// ////////////////////
			// initialize node set
			this.m_nNodes = new int[separator.size()];
			int iPos = 0;
			this.m_nCardinality = 1;
			for (Integer element : separator) {
				int iNode = element;
				this.m_nNodes[iPos++] = iNode;
				this.m_nCardinality *= bayesNet.getCardinality(iNode);
			}
			this.m_parentNode = parentNode;
			this.m_childNode = childNode;
			this.m_bayesNet = bayesNet;
		} // c'tor

		/**
		 * marginalize junciontTreeNode node over all nodes outside the separator set of the parent clique
		 *
		 */
		public void updateFromParent() {
			double[] fis = this.update(this.m_parentNode);
			if (fis == null) {
				this.m_fiParent = null;
			} else {
				this.m_fiParent = fis;
				// normalize
				double sum = 0;
				for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
					sum += this.m_fiParent[iPos];
				}
				for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
					this.m_fiParent[iPos] /= sum;
				}
			}
		} // updateFromParent

		/**
		 * marginalize junciontTreeNode node over all nodes outside the separator set of the child clique
		 *
		 */
		public void updateFromChild() {
			double[] fis = this.update(this.m_childNode);
			if (fis == null) {
				this.m_fiChild = null;
			} else {
				this.m_fiChild = fis;
				// normalize
				double sum = 0;
				for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
					sum += this.m_fiChild[iPos];
				}
				for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
					this.m_fiChild[iPos] /= sum;
				}
			}
		} // updateFromChild

		/**
		 * marginalize junciontTreeNode node over all nodes outside the separator set
		 *
		 * @param node
		 *            one of the neighboring junciont tree nodes of this separator
		 */
		public double[] update(final JunctionTreeNode node) {
			if (node.m_P == null) {
				return null;
			}
			double[] fi = new double[this.m_nCardinality];

			int[] values = new int[node.m_nNodes.length];
			int[] order = new int[this.m_bayesNet.getNrOfNodes()];
			for (int iNode = 0; iNode < node.m_nNodes.length; iNode++) {
				order[node.m_nNodes[iNode]] = iNode;
			}
			// fill in the values
			for (int iPos = 0; iPos < node.m_nCardinality; iPos++) {
				int iNodeCPT = MarginCalculator.this.getCPT(node.m_nNodes, node.m_nNodes.length, values, order, this.m_bayesNet);
				int iSepCPT = MarginCalculator.this.getCPT(this.m_nNodes, this.m_nNodes.length, values, order, this.m_bayesNet);
				fi[iSepCPT] += node.m_P[iNodeCPT];
				// update values
				int i = 0;
				values[i]++;
				while (i < node.m_nNodes.length && values[i] == this.m_bayesNet.getCardinality(node.m_nNodes[i])) {
					values[i] = 0;
					i++;
					if (i < node.m_nNodes.length) {
						values[i]++;
					}
				}
			}
			return fi;
		} // update

		/**
		 * Returns the revision string.
		 *
		 * @return the revision
		 */
		@Override
		public String getRevision() {
			return RevisionUtils.extract("$Revision$");
		}

	} // class JunctionTreeSeparator

	public class JunctionTreeNode implements Serializable, RevisionHandler {

		private static final long serialVersionUID = 650278019241175536L;
		/**
		 * reference Bayes net for information about variables like name, cardinality, etc. but not for relations between nodes
		 **/
		BayesNet m_bayesNet;
		/** nodes of the Bayes net in this junction node **/
		public int[] m_nNodes;
		/** cardinality of the instances of variables in this junction node **/
		int m_nCardinality;
		/** potentials for first network **/
		double[] m_fi;

		/** distribution over this junction node according to first Bayes network **/
		double[] m_P;

		double[][] m_MarginalP;

		JunctionTreeSeparator m_parentSeparator;

		public void setParentSeparator(final JunctionTreeSeparator parentSeparator) {
			this.m_parentSeparator = parentSeparator;
		}

		public Vector<JunctionTreeNode> m_children;

		public void addChildClique(final JunctionTreeNode child) {
			this.m_children.add(child);
		}

		public void initializeUp() {
			this.m_P = new double[this.m_nCardinality];
			for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
				this.m_P[iPos] = this.m_fi[iPos];
			}
			int[] values = new int[this.m_nNodes.length];
			int[] order = new int[this.m_bayesNet.getNrOfNodes()];
			for (int iNode = 0; iNode < this.m_nNodes.length; iNode++) {
				order[this.m_nNodes[iNode]] = iNode;
			}
			for (JunctionTreeNode element : this.m_children) {
				JunctionTreeNode childNode = element;
				JunctionTreeSeparator separator = childNode.m_parentSeparator;
				// Update the values
				for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
					int iSepCPT = MarginCalculator.this.getCPT(separator.m_nNodes, separator.m_nNodes.length, values, order, this.m_bayesNet);
					int iNodeCPT = MarginCalculator.this.getCPT(this.m_nNodes, this.m_nNodes.length, values, order, this.m_bayesNet);
					this.m_P[iNodeCPT] *= separator.m_fiChild[iSepCPT];
					// update values
					int i = 0;
					values[i]++;
					while (i < this.m_nNodes.length && values[i] == this.m_bayesNet.getCardinality(this.m_nNodes[i])) {
						values[i] = 0;
						i++;
						if (i < this.m_nNodes.length) {
							values[i]++;
						}
					}
				}
			}
			// normalize
			double sum = 0;
			for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
				sum += this.m_P[iPos];
			}
			for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
				this.m_P[iPos] /= sum;
			}

			if (this.m_parentSeparator != null) { // not a root node
				this.m_parentSeparator.updateFromChild();
			}
		} // initializeUp

		public void initializeDown(final boolean recursively) {
			if (this.m_parentSeparator == null) { // a root node
				this.calcMarginalProbabilities();
			} else {
				this.m_parentSeparator.updateFromParent();
				int[] values = new int[this.m_nNodes.length];
				int[] order = new int[this.m_bayesNet.getNrOfNodes()];
				for (int iNode = 0; iNode < this.m_nNodes.length; iNode++) {
					order[this.m_nNodes[iNode]] = iNode;
				}

				// Update the values
				for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
					int iSepCPT = MarginCalculator.this.getCPT(this.m_parentSeparator.m_nNodes, this.m_parentSeparator.m_nNodes.length, values, order, this.m_bayesNet);
					int iNodeCPT = MarginCalculator.this.getCPT(this.m_nNodes, this.m_nNodes.length, values, order, this.m_bayesNet);
					if (this.m_parentSeparator.m_fiChild[iSepCPT] > 0) {
						this.m_P[iNodeCPT] *= this.m_parentSeparator.m_fiParent[iSepCPT] / this.m_parentSeparator.m_fiChild[iSepCPT];
					} else {
						this.m_P[iNodeCPT] = 0;
					}
					// update values
					int i = 0;
					values[i]++;
					while (i < this.m_nNodes.length && values[i] == this.m_bayesNet.getCardinality(this.m_nNodes[i])) {
						values[i] = 0;
						i++;
						if (i < this.m_nNodes.length) {
							values[i]++;
						}
					}
				}
				// normalize
				double sum = 0;
				for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
					sum += this.m_P[iPos];
				}
				for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
					this.m_P[iPos] /= sum;
				}
				this.m_parentSeparator.updateFromChild();
				this.calcMarginalProbabilities();
			}
			if (recursively) {
				for (Object element : this.m_children) {
					JunctionTreeNode childNode = (JunctionTreeNode) element;
					childNode.initializeDown(true);
				}
			}
		} // initializeDown

		/**
		 * calculate marginal probabilities for the individual nodes in the clique. Store results in m_MarginalP
		 */
		void calcMarginalProbabilities() {
			// calculate marginal probabilities
			int[] values = new int[this.m_nNodes.length];
			int[] order = new int[this.m_bayesNet.getNrOfNodes()];
			this.m_MarginalP = new double[this.m_nNodes.length][];
			for (int iNode = 0; iNode < this.m_nNodes.length; iNode++) {
				order[this.m_nNodes[iNode]] = iNode;
				this.m_MarginalP[iNode] = new double[this.m_bayesNet.getCardinality(this.m_nNodes[iNode])];
			}
			for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
				int iNodeCPT = MarginCalculator.this.getCPT(this.m_nNodes, this.m_nNodes.length, values, order, this.m_bayesNet);
				for (int iNode = 0; iNode < this.m_nNodes.length; iNode++) {
					this.m_MarginalP[iNode][values[iNode]] += this.m_P[iNodeCPT];
				}
				// update values
				int i = 0;
				values[i]++;
				while (i < this.m_nNodes.length && values[i] == this.m_bayesNet.getCardinality(this.m_nNodes[i])) {
					values[i] = 0;
					i++;
					if (i < this.m_nNodes.length) {
						values[i]++;
					}
				}
			}

			for (int iNode = 0; iNode < this.m_nNodes.length; iNode++) {
				MarginCalculator.this.m_Margins[this.m_nNodes[iNode]] = this.m_MarginalP[iNode];
			}
		} // calcMarginalProbabilities

		@Override
		public String toString() {
			StringBuffer buf = new StringBuffer();
			for (int iNode = 0; iNode < this.m_nNodes.length; iNode++) {
				buf.append(this.m_bayesNet.getNodeName(this.m_nNodes[iNode]) + ": ");
				for (int iValue = 0; iValue < this.m_MarginalP[iNode].length; iValue++) {
					buf.append(this.m_MarginalP[iNode][iValue] + " ");
				}
				buf.append('\n');
			}
			for (Object element : this.m_children) {
				JunctionTreeNode childNode = (JunctionTreeNode) element;
				buf.append("----------------\n");
				buf.append(childNode.toString());
			}
			return buf.toString();
		} // toString

		void calculatePotentials(final BayesNet bayesNet, final Set<Integer> clique, final boolean[] bDone) {
			this.m_fi = new double[this.m_nCardinality];

			int[] values = new int[this.m_nNodes.length];
			int[] order = new int[bayesNet.getNrOfNodes()];
			for (int iNode = 0; iNode < this.m_nNodes.length; iNode++) {
				order[this.m_nNodes[iNode]] = iNode;
			}
			// find conditional probabilities that need to be taken in account
			boolean[] bIsContained = new boolean[this.m_nNodes.length];
			for (int iNode = 0; iNode < this.m_nNodes.length; iNode++) {
				int nNode = this.m_nNodes[iNode];
				bIsContained[iNode] = !bDone[nNode];
				for (int iParent = 0; iParent < bayesNet.getNrOfParents(nNode); iParent++) {
					int nParent = bayesNet.getParent(nNode, iParent);
					if (!clique.contains(nParent)) {
						bIsContained[iNode] = false;
					}
				}
				if (bIsContained[iNode]) {
					bDone[nNode] = true;
					if (MarginCalculator.this.m_debug) {
						System.out.println("adding node " + nNode);
					}
				}
			}

			// fill in the values
			for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
				int iCPT = MarginCalculator.this.getCPT(this.m_nNodes, this.m_nNodes.length, values, order, bayesNet);
				this.m_fi[iCPT] = 1.0;
				for (int iNode = 0; iNode < this.m_nNodes.length; iNode++) {
					if (bIsContained[iNode]) {
						int nNode = this.m_nNodes[iNode];
						int[] nNodes = bayesNet.getParentSet(nNode).getParents();
						int iCPT2 = MarginCalculator.this.getCPT(nNodes, bayesNet.getNrOfParents(nNode), values, order, bayesNet);
						double f = bayesNet.getDistributions()[nNode][iCPT2].getProbability(values[iNode]);
						this.m_fi[iCPT] *= f;
					}
				}

				// update values
				int i = 0;
				values[i]++;
				while (i < this.m_nNodes.length && values[i] == bayesNet.getCardinality(this.m_nNodes[i])) {
					values[i] = 0;
					i++;
					if (i < this.m_nNodes.length) {
						values[i]++;
					}
				}
			}
		} // calculatePotentials

		JunctionTreeNode(final Set<Integer> clique, final BayesNet bayesNet, final boolean[] bDone) {
			this.m_bayesNet = bayesNet;
			this.m_children = new Vector<JunctionTreeNode>();
			// ////////////////////
			// initialize node set
			this.m_nNodes = new int[clique.size()];
			int iPos = 0;
			this.m_nCardinality = 1;
			for (Integer integer : clique) {
				int iNode = integer;
				this.m_nNodes[iPos++] = iNode;
				this.m_nCardinality *= bayesNet.getCardinality(iNode);
			}
			// //////////////////////////////
			// initialize potential function
			this.calculatePotentials(bayesNet, clique, bDone);
		} // JunctionTreeNode c'tor

		/*
		 * check whether this junciton tree node contains node nNode
		 */
		boolean contains(final int nNode) {
			for (int m_nNode : this.m_nNodes) {
				if (m_nNode == nNode) {
					return true;
				}
			}
			return false;
		} // contains

		public void setEvidence(final int nNode, final int iValue) throws Exception {
			int[] values = new int[this.m_nNodes.length];
			int[] order = new int[this.m_bayesNet.getNrOfNodes()];

			int nNodeIdx = -1;
			for (int iNode = 0; iNode < this.m_nNodes.length; iNode++) {
				order[this.m_nNodes[iNode]] = iNode;
				if (this.m_nNodes[iNode] == nNode) {
					nNodeIdx = iNode;
				}
			}
			if (nNodeIdx < 0) {
				throw new Exception("setEvidence: Node " + nNode + " not found in this clique");
			}
			for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
				if (values[nNodeIdx] != iValue) {
					int iNodeCPT = MarginCalculator.this.getCPT(this.m_nNodes, this.m_nNodes.length, values, order, this.m_bayesNet);
					this.m_P[iNodeCPT] = 0;
				}
				// update values
				int i = 0;
				values[i]++;
				while (i < this.m_nNodes.length && values[i] == this.m_bayesNet.getCardinality(this.m_nNodes[i])) {
					values[i] = 0;
					i++;
					if (i < this.m_nNodes.length) {
						values[i]++;
					}
				}
			}
			// normalize
			double sum = 0;
			for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
				sum += this.m_P[iPos];
			}
			for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
				this.m_P[iPos] /= sum;
			}
			this.calcMarginalProbabilities();
			this.updateEvidence(this);
		} // setEvidence

		void updateEvidence(final JunctionTreeNode source) {
			if (source != this) {
				int[] values = new int[this.m_nNodes.length];
				int[] order = new int[this.m_bayesNet.getNrOfNodes()];
				for (int iNode = 0; iNode < this.m_nNodes.length; iNode++) {
					order[this.m_nNodes[iNode]] = iNode;
				}
				int[] nChildNodes = source.m_parentSeparator.m_nNodes;
				int nNumChildNodes = nChildNodes.length;
				for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
					int iNodeCPT = MarginCalculator.this.getCPT(this.m_nNodes, this.m_nNodes.length, values, order, this.m_bayesNet);
					int iChildCPT = MarginCalculator.this.getCPT(nChildNodes, nNumChildNodes, values, order, this.m_bayesNet);
					if (source.m_parentSeparator.m_fiParent[iChildCPT] != 0) {
						this.m_P[iNodeCPT] *= source.m_parentSeparator.m_fiChild[iChildCPT] / source.m_parentSeparator.m_fiParent[iChildCPT];
					} else {
						this.m_P[iNodeCPT] = 0;
					}
					// update values
					int i = 0;
					values[i]++;
					while (i < this.m_nNodes.length && values[i] == this.m_bayesNet.getCardinality(this.m_nNodes[i])) {
						values[i] = 0;
						i++;
						if (i < this.m_nNodes.length) {
							values[i]++;
						}
					}
				}
				// normalize
				double sum = 0;
				for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
					sum += this.m_P[iPos];
				}
				for (int iPos = 0; iPos < this.m_nCardinality; iPos++) {
					this.m_P[iPos] /= sum;
				}
				this.calcMarginalProbabilities();
			}
			for (Object element : this.m_children) {
				JunctionTreeNode childNode = (JunctionTreeNode) element;
				if (childNode != source) {
					childNode.initializeDown(true);
				}
			}
			if (this.m_parentSeparator != null) {
				this.m_parentSeparator.updateFromChild();
				this.m_parentSeparator.m_parentNode.updateEvidence(this);
				this.m_parentSeparator.updateFromParent();
			}
		} // updateEvidence

		/**
		 * Returns the revision string.
		 *
		 * @return the revision
		 */
		@Override
		public String getRevision() {
			return RevisionUtils.extract("$Revision$");
		}

	} // class JunctionTreeNode

	int getCPT(final int[] nodeSet, final int nNodes, final int[] values, final int[] order, final BayesNet bayesNet) {
		int iCPTnew = 0;
		for (int iNode = 0; iNode < nNodes; iNode++) {
			int nNode = nodeSet[iNode];
			iCPTnew = iCPTnew * bayesNet.getCardinality(nNode);
			iCPTnew += values[order[nNode]];
		}
		return iCPTnew;
	} // getCPT

	int[] getCliqueTree(final int[] order, final Set<Integer>[] cliques, final Set<Integer>[] separators) {
		int nNodes = order.length;
		int[] parentCliques = new int[nNodes];
		// for (int i = nNodes - 1; i >= 0; i--) {
		for (int i = 0; i < nNodes; i++) {
			int iNode = order[i];
			parentCliques[iNode] = -1;
			if (cliques[iNode] != null && separators[iNode].size() > 0) {
				// for (int j = nNodes - 1; j > i; j--) {
				for (int j = 0; j < nNodes; j++) {
					int iNode2 = order[j];
					if (iNode != iNode2 && cliques[iNode2] != null && cliques[iNode2].containsAll(separators[iNode])) {
						parentCliques[iNode] = iNode2;
						j = i;
						j = 0;
						j = nNodes;
					}
				}

			}
		}
		return parentCliques;
	} // getCliqueTree

	/**
	 * calculate separator sets in clique tree
	 *
	 * @param order:
	 *            maximum cardinality ordering of the graph
	 * @param cliques:
	 *            set of cliques
	 * @return set of separator sets
	 */
	Set<Integer>[] getSeparators(final int[] order, final Set<Integer>[] cliques) {
		int nNodes = order.length;
		@SuppressWarnings("unchecked")
		Set<Integer>[] separators = new HashSet[nNodes];
		Set<Integer> processedNodes = new HashSet<Integer>();
		// for (int i = nNodes - 1; i >= 0; i--) {
		for (int i = 0; i < nNodes; i++) {
			int iNode = order[i];
			if (cliques[iNode] != null) {
				Set<Integer> separator = new HashSet<Integer>();
				separator.addAll(cliques[iNode]);
				separator.retainAll(processedNodes);
				separators[iNode] = separator;
				processedNodes.addAll(cliques[iNode]);
			}
		}
		return separators;
	} // getSeparators

	/**
	 * get cliques in a decomposable graph represented by an adjacency matrix
	 *
	 * @param order:
	 *            maximum cardinality ordering of the graph
	 * @param bAdjacencyMatrix:
	 *            decomposable graph
	 * @return set of cliques
	 */
	Set<Integer>[] getCliques(final int[] order, final boolean[][] bAdjacencyMatrix) throws Exception {
		int nNodes = bAdjacencyMatrix.length;
		@SuppressWarnings("unchecked")
		Set<Integer>[] cliques = new HashSet[nNodes];
		// int[] inverseOrder = new int[nNodes];
		// for (int iNode = 0; iNode < nNodes; iNode++) {
		// inverseOrder[order[iNode]] = iNode;
		// }
		// consult nodes in reverse order
		for (int i = nNodes - 1; i >= 0; i--) {
			int iNode = order[i];
			if (iNode == 22) {
			}
			Set<Integer> clique = new HashSet<Integer>();
			clique.add(iNode);
			for (int j = 0; j < i; j++) {
				int iNode2 = order[j];
				if (bAdjacencyMatrix[iNode][iNode2]) {
					clique.add(iNode2);
				}
			}

			// for (int iNode2 = 0; iNode2 < nNodes; iNode2++) {
			// if (bAdjacencyMatrix[iNode][iNode2] && inverseOrder[iNode2] <
			// inverseOrder[iNode]) {
			// clique.add(iNode2);
			// }
			// }
			cliques[iNode] = clique;
		}
		for (int iNode = 0; iNode < nNodes; iNode++) {
			for (int iNode2 = 0; iNode2 < nNodes; iNode2++) {
				if (iNode != iNode2 && cliques[iNode] != null && cliques[iNode2] != null && cliques[iNode].containsAll(cliques[iNode2])) {
					cliques[iNode2] = null;
				}
			}
		}
		// sanity check
		if (this.m_debug) {
			int[] nNodeSet = new int[nNodes];
			for (int iNode = 0; iNode < nNodes; iNode++) {
				if (cliques[iNode] != null) {
					Iterator<Integer> it = cliques[iNode].iterator();
					int k = 0;
					while (it.hasNext()) {
						nNodeSet[k++] = it.next();
					}
					for (int i = 0; i < cliques[iNode].size(); i++) {
						for (int j = 0; j < cliques[iNode].size(); j++) {
							if (i != j && !bAdjacencyMatrix[nNodeSet[i]][nNodeSet[j]]) {
								throw new Exception("Non clique" + i + " " + j);
							}
						}
					}
				}
			}
		}
		return cliques;
	} // getCliques

	/**
	 * moralize DAG and calculate adjacency matrix representation for a Bayes Network, effecively converting the directed acyclic graph to an undirected graph.
	 *
	 * @param bayesNet
	 *            Bayes Network to process
	 * @return adjacencies in boolean matrix format
	 */
	public boolean[][] moralize(final BayesNet bayesNet) {
		int nNodes = bayesNet.getNrOfNodes();
		boolean[][] bAdjacencyMatrix = new boolean[nNodes][nNodes];
		for (int iNode = 0; iNode < nNodes; iNode++) {
			ParentSet parents = bayesNet.getParentSets()[iNode];
			this.moralizeNode(parents, iNode, bAdjacencyMatrix);
		}
		return bAdjacencyMatrix;
	} // moralize

	private void moralizeNode(final ParentSet parents, final int iNode, final boolean[][] bAdjacencyMatrix) {
		for (int iParent = 0; iParent < parents.getNrOfParents(); iParent++) {
			int nParent = parents.getParent(iParent);
			if (this.m_debug && !bAdjacencyMatrix[iNode][nParent]) {
				System.out.println("Insert " + iNode + "--" + nParent);
			}
			bAdjacencyMatrix[iNode][nParent] = true;
			bAdjacencyMatrix[nParent][iNode] = true;
			for (int iParent2 = iParent + 1; iParent2 < parents.getNrOfParents(); iParent2++) {
				int nParent2 = parents.getParent(iParent2);
				if (this.m_debug && !bAdjacencyMatrix[nParent2][nParent]) {
					System.out.println("Mary " + nParent + "--" + nParent2);
				}
				bAdjacencyMatrix[nParent2][nParent] = true;
				bAdjacencyMatrix[nParent][nParent2] = true;
			}
		}
	} // moralizeNode

	/**
	 * Apply Tarjan and Yannakakis (1984) fill in algorithm for graph triangulation. In reverse order, insert edges between any non-adjacent neighbors that are lower numbered in the ordering.
	 *
	 * Side effect: input matrix is used as output
	 *
	 * @param order
	 *            node ordering
	 * @param bAdjacencyMatrix
	 *            boolean matrix representing the graph
	 * @return boolean matrix representing the graph with fill ins
	 */
	public boolean[][] fillIn(final int[] order, final boolean[][] bAdjacencyMatrix) {
		int nNodes = bAdjacencyMatrix.length;
		int[] inverseOrder = new int[nNodes];
		for (int iNode = 0; iNode < nNodes; iNode++) {
			inverseOrder[order[iNode]] = iNode;
		}
		// consult nodes in reverse order
		for (int i = nNodes - 1; i >= 0; i--) {
			int iNode = order[i];
			// find pairs of neighbors with lower order
			for (int j = 0; j < i; j++) {
				int iNode2 = order[j];
				if (bAdjacencyMatrix[iNode][iNode2]) {
					for (int k = j + 1; k < i; k++) {
						int iNode3 = order[k];
						if (bAdjacencyMatrix[iNode][iNode3]) {
							// fill in
							if (this.m_debug && (!bAdjacencyMatrix[iNode2][iNode3] || !bAdjacencyMatrix[iNode3][iNode2])) {
								System.out.println("Fill in " + iNode2 + "--" + iNode3);
							}
							bAdjacencyMatrix[iNode2][iNode3] = true;
							bAdjacencyMatrix[iNode3][iNode2] = true;
						}
					}
				}
			}
		}
		return bAdjacencyMatrix;
	} // fillIn

	/**
	 * calculate maximum cardinality ordering; start with first node add node that has most neighbors already ordered till all nodes are in the ordering
	 *
	 * This implementation does not assume the graph is connected
	 *
	 * @param bAdjacencyMatrix:
	 *            n by n matrix with adjacencies in graph of n nodes
	 * @return maximum cardinality ordering
	 * @throws InterruptedException
	 */
	int[] getMaxCardOrder(final boolean[][] bAdjacencyMatrix) throws InterruptedException {
		int nNodes = bAdjacencyMatrix.length;
		int[] order = new int[nNodes];
		if (nNodes == 0) {
			return order;
		}
		boolean[] bDone = new boolean[nNodes];
		// start with node 0
		order[0] = 0;
		bDone[0] = true;
		// order remaining nodes
		for (int iNode = 1; iNode < nNodes; iNode++) {
			// XXX kill weka
			if (Thread.currentThread().isInterrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			int nMaxCard = -1;
			int iBestNode = -1;
			// find node with higest cardinality of previously ordered nodes
			for (int iNode2 = 0; iNode2 < nNodes; iNode2++) {
				if (!bDone[iNode2]) {
					int nCard = 0;
					// calculate cardinality for node iNode2
					for (int iNode3 = 0; iNode3 < nNodes; iNode3++) {
						if (bAdjacencyMatrix[iNode2][iNode3] && bDone[iNode3]) {
							nCard++;
						}
					}
					if (nCard > nMaxCard) {
						nMaxCard = nCard;
						iBestNode = iNode2;
					}
				}
			}
			order[iNode] = iBestNode;
			bDone[iBestNode] = true;
		}
		return order;
	} // getMaxCardOrder

	public void setEvidence(final int nNode, final int iValue) throws Exception {
		if (this.m_root == null) {
			throw new Exception("Junction tree not initialize yet");
		}
		int iJtNode = 0;
		while (iJtNode < this.jtNodes.length && (this.jtNodes[iJtNode] == null || !this.jtNodes[iJtNode].contains(nNode))) {
			iJtNode++;
		}
		if (this.jtNodes.length == iJtNode) {
			throw new Exception("Could not find node " + nNode + " in junction tree");
		}
		this.jtNodes[iJtNode].setEvidence(nNode, iValue);
	} // setEvidence

	@Override
	public String toString() {
		return this.m_root.toString();
	} // toString

	double[][] m_Margins;

	public double[] getMargin(final int iNode) {
		return this.m_Margins[iNode];
	} // getMargin

	/**
	 * Returns the revision string.
	 *
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision$");
	}

	public static void main(final String[] args) {
		try {
			BIFReader bayesNet = new BIFReader();
			bayesNet.processFile(args[0]);

			MarginCalculator dc = new MarginCalculator();
			dc.calcMargins(bayesNet);
			int iNode = 2;
			int iValue = 0;
			int iNode2 = 4;
			int iValue2 = 0;
			dc.setEvidence(iNode, iValue);
			dc.setEvidence(iNode2, iValue2);
			System.out.print(dc.toString());

			dc.calcFullMargins(bayesNet);
			dc.setEvidence(iNode, iValue);
			dc.setEvidence(iNode2, iValue2);
			System.out.println("==============");
			System.out.print(dc.toString());

		} catch (Exception e) {
			e.printStackTrace();
		}
	} // main

} // class MarginCalculator
