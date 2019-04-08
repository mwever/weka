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
 *    HNode.java
 *    Copyright (C) 2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.ht;

import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.Map;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Utils;

/**
 * Abstract base class for nodes in a Hoeffding tree
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @revision $Revision$
 */
public abstract class HNode implements Serializable {
	/**
	 * For serialization
	 */
	private static final long serialVersionUID = 197233928177240264L;

	/** Class distribution at this node */
	public Map<String, WeightMass> m_classDistribution = new LinkedHashMap<>();

	/** Holds the leaf number (if this is a leaf) */
	protected int m_leafNum;

	/** Holds the node number (for graphing purposes) */
	protected int m_nodeNum;

	/**
	 * Construct a new HNode
	 */
	public HNode() {
	}

	/**
	 * Construct a new HNode with the supplied class distribution
	 *
	 * @param classDistrib
	 */
	public HNode(final Map<String, WeightMass> classDistrib) {
		this.m_classDistribution = classDistrib;
	}

	/**
	 * Returns true if this is a leaf
	 *
	 * @return
	 */
	public boolean isLeaf() {
		return true;
	}

	/**
	 * The size of the class distribution
	 *
	 * @return the number of entries in the class distribution
	 */
	public int numEntriesInClassDistribution() {
		return this.m_classDistribution.size();
	}

	/**
	 * Returns true if the class distribution is pure
	 *
	 * @return true if the class distribution is pure
	 */
	public boolean classDistributionIsPure() {
		int count = 0;
		for (Map.Entry<String, WeightMass> el : this.m_classDistribution.entrySet()) {
			if (el.getValue().m_weight > 0) {
				count++;

				if (count > 1) {
					break;
				}
			}
		}

		return (count < 2);
	}

	/**
	 * Update the class frequency distribution with the supplied instance
	 *
	 * @param inst
	 *          the instance to update with
	 */
	public void updateDistribution(final Instance inst) {
		if (inst.classIsMissing()) {
			return;
		}
		String classVal = inst.stringValue(inst.classAttribute());

		WeightMass m = this.m_classDistribution.get(classVal);
		if (m == null) {
			m = new WeightMass();
			m.m_weight = 1.0;

			this.m_classDistribution.put(classVal, m);
		}
		m.m_weight += inst.weight();
	}

	/**
	 * Return a class probability distribution computed from the frequency counts at this node
	 *
	 * @param inst
	 *          the instance to get a prediction for
	 * @param classAtt
	 *          the class attribute
	 * @return a class probability distribution
	 * @throws Exception
	 *           if a problem occurs
	 */
	public double[] getDistribution(final Instance inst, final Attribute classAtt) throws Exception {
		double[] dist = new double[classAtt.numValues()];

		for (int i = 0; i < classAtt.numValues(); i++) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			WeightMass w = this.m_classDistribution.get(classAtt.value(i));
			if (w != null) {
				dist[i] = w.m_weight;
			} else {
				dist[i] = 1.0;
			}
		}

		Utils.normalize(dist);
		return dist;
	}

	public int installNodeNums(int nodeNum) {
		nodeNum++;
		this.m_nodeNum = nodeNum;

		return nodeNum;
	}

	protected int dumpTree(final int depth, int leafCount, final StringBuffer buff) {

		double max = -1;
		String classVal = "";
		for (Map.Entry<String, WeightMass> e : this.m_classDistribution.entrySet()) {
			if (e.getValue().m_weight > max) {
				max = e.getValue().m_weight;
				classVal = e.getKey();
			}
		}
		buff.append(classVal + " (" + String.format("%-9.3f", max).trim() + ")");
		leafCount++;
		this.m_leafNum = leafCount;

		return leafCount;
	}

	protected void printLeafModels(final StringBuffer buff) {
	}

	public void graphTree(final StringBuffer text) {

		double max = -1;
		String classVal = "";
		for (Map.Entry<String, WeightMass> e : this.m_classDistribution.entrySet()) {
			if (e.getValue().m_weight > max) {
				max = e.getValue().m_weight;
				classVal = e.getKey();
			}
		}

		text.append("N" + this.m_nodeNum + " [label=\"" + classVal + " (" + String.format("%-9.3f", max).trim() + ")\" shape=box style=filled]\n");
	}

	/**
	 * Print a textual description of the tree
	 *
	 * @param printLeaf
	 *          true if leaf models (NB, NB adaptive) should be output
	 * @return a textual description of the tree
	 */
	public String toString(final boolean printLeaf) {

		this.installNodeNums(0);

		StringBuffer buff = new StringBuffer();

		this.dumpTree(0, 0, buff);

		if (printLeaf) {
			buff.append("\n\n");
			this.printLeafModels(buff);
		}

		return buff.toString();
	}

	/**
	 * Return the total weight of instances seen at this node
	 *
	 * @return the total weight of instances seen at this node
	 */
	public double totalWeight() {
		double tw = 0;

		for (Map.Entry<String, WeightMass> e : this.m_classDistribution.entrySet()) {
			tw += e.getValue().m_weight;
		}

		return tw;
	}

	/**
	 * Return the leaf that the supplied instance ends up at
	 *
	 * @param inst
	 *          the instance to find the leaf for
	 * @param parent
	 *          the parent node
	 * @param parentBranch
	 *          the parent branch
	 * @return the leaf that the supplied instance ends up at
	 */
	public LeafNode leafForInstance(final Instance inst, final SplitNode parent, final String parentBranch) {
		return new LeafNode(this, parent, parentBranch);
	}

	/**
	 * Update the node with the supplied instance
	 *
	 * @param inst
	 *          the instance to update with
	 * @throws Exception
	 *           if a problem occurs
	 */
	public abstract void updateNode(Instance inst) throws Exception;
}
