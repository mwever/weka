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
 *    Rule.java
 *    Copyright (C) 2000-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.m5;

import java.io.Serializable;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * Generates a single m5 tree or rule
 *
 * @author Mark Hall
 * @version $Revision$
 */
public class Rule implements Serializable, RevisionHandler {

	/** for serialization */
	private static final long serialVersionUID = -4458627451682483204L;

	protected static int LEFT = 0;
	protected static int RIGHT = 1;

	/**
	 * the instances covered by this rule
	 */
	private Instances m_instances;

	/**
	 * the class index
	 */
	private int m_classIndex;

	/**
	 * the number of instances in the dataset
	 */
	private int m_numInstances;

	/**
	 * the indexes of the attributes used to split on for this rule
	 */
	private int[] m_splitAtts;

	/**
	 * the corresponding values of the split points
	 */
	private double[] m_splitVals;

	/**
	 * the corresponding internal nodes. Used for smoothing rules.
	 */
	private RuleNode[] m_internalNodes;

	/**
	 * the corresponding relational operators (0 = "<=", 1 = ">")
	 */
	private int[] m_relOps;

	/**
	 * the leaf encapsulating the linear model for this rule
	 */
	private RuleNode m_ruleModel;

	/**
	 * the top of the m5 tree for this rule
	 */
	protected RuleNode m_topOfTree;

	/**
	 * the standard deviation of the class for all the instances
	 */
	private double m_globalStdDev;

	/**
	 * the absolute deviation of the class for all the instances
	 */
	private double m_globalAbsDev;

	/**
	 * the instances covered by this rule
	 */
	private Instances m_covered;

	/**
	 * the number of instances covered by this rule
	 */
	private int m_numCovered;

	/**
	 * the instances not covered by this rule
	 */
	private Instances m_notCovered;

	/**
	 * use a pruned m5 tree rather than make a rule
	 */
	private boolean m_useTree;

	/**
	 * use the original m5 smoothing procedure
	 */
	private boolean m_smoothPredictions;

	/**
	 * Save instances at each node in an M5 tree for visualization purposes.
	 */
	private boolean m_saveInstances;

	/**
	 * Make a regression tree instead of a model tree
	 */
	private boolean m_regressionTree;

	/**
	 * Build unpruned tree/rule
	 */
	private boolean m_useUnpruned;

	/**
	 * The minimum number of instances to allow at a leaf node
	 */
	private double m_minNumInstances;

	/**
	 * Constructor declaration
	 * 
	 */
	public Rule() {
		this.m_useTree = false;
		this.m_smoothPredictions = false;
		this.m_useUnpruned = false;
		this.m_minNumInstances = 4;
	}

	/**
	 * Generates a single rule or m5 model tree.
	 * 
	 * @param data
	 *            set of instances serving as training data
	 * @exception Exception
	 *                if the rule has not been generated successfully
	 */
	public void buildClassifier(final Instances data) throws Exception {
		this.m_instances = null;
		this.m_topOfTree = null;
		this.m_covered = null;
		this.m_notCovered = null;
		this.m_ruleModel = null;
		this.m_splitAtts = null;
		this.m_splitVals = null;
		this.m_relOps = null;
		this.m_internalNodes = null;
		this.m_instances = data;
		this.m_classIndex = this.m_instances.classIndex();
		this.m_numInstances = this.m_instances.numInstances();

		// first calculate global deviation of class attribute
		this.m_globalStdDev = Rule.stdDev(this.m_classIndex, this.m_instances);
		this.m_globalAbsDev = Rule.absDev(this.m_classIndex, this.m_instances);

		this.m_topOfTree = new RuleNode(this.m_globalStdDev, this.m_globalAbsDev, null);
		this.m_topOfTree.setSaveInstances(this.m_saveInstances);
		this.m_topOfTree.setRegressionTree(this.m_regressionTree);
		this.m_topOfTree.setMinNumInstances(this.m_minNumInstances);
		this.m_topOfTree.buildClassifier(this.m_instances);

		if (!this.m_useUnpruned) {
			this.m_topOfTree.prune();
		} else {
			this.m_topOfTree.installLinearModels();
		}

		if (this.m_smoothPredictions) {
			this.m_topOfTree.installSmoothedModels();
		}
		// m_topOfTree.printAllModels();
		this.m_topOfTree.numLeaves(0);

		if (!this.m_useTree) {
			this.makeRule();
			// save space
			// m_topOfTree = null;
		}

		// save space
		this.m_instances = new Instances(this.m_instances, 0);

	}

	/**
	 * Calculates a prediction for an instance using this rule or M5 model tree
	 * 
	 * @param instance
	 *            the instance whos class value is to be predicted
	 * @return the prediction
	 * @exception Exception
	 *                if a prediction can't be made.
	 */
	public double classifyInstance(final Instance instance) throws Exception {
		if (this.m_useTree) {
			return this.m_topOfTree.classifyInstance(instance);
		}

		// does the instance pass the rule's conditions?
		if (this.m_splitAtts.length > 0) {
			for (int i = 0; i < this.m_relOps.length; i++) {
				if (this.m_relOps[i] == LEFT) // left
				{
					if (instance.value(this.m_splitAtts[i]) > this.m_splitVals[i]) {
						throw new Exception("Rule does not classify instance");
					}
				} else {
					if (instance.value(this.m_splitAtts[i]) <= this.m_splitVals[i]) {
						throw new Exception("Rule does not classify instance");
					}
				}
			}
		}

		// the linear model's prediction for this rule
		return this.m_ruleModel.classifyInstance(instance);
	}

	/**
	 * Returns the top of the tree.
	 */
	public RuleNode topOfTree() {

		return this.m_topOfTree;
	}

	/**
	 * Make the single best rule from a pruned m5 model tree
	 * 
	 * @exception Exception
	 *                if something goes wrong.
	 */
	private void makeRule() throws Exception {
		RuleNode[] best_leaf = new RuleNode[1];
		double[] best_cov = new double[1];
		RuleNode temp;

		this.m_notCovered = new Instances(this.m_instances, 0);
		this.m_covered = new Instances(this.m_instances, 0);
		best_cov[0] = -1;
		best_leaf[0] = null;

		this.m_topOfTree.findBestLeaf(best_cov, best_leaf);

		temp = best_leaf[0];

		if (temp == null) {
			throw new Exception("Unable to generate rule!");
		}

		// save the linear model for this rule
		this.m_ruleModel = temp;

		int count = 0;

		while (temp.parentNode() != null) {
			count++;
			temp = temp.parentNode();
		}

		temp = best_leaf[0];
		this.m_relOps = new int[count];
		this.m_splitAtts = new int[count];
		this.m_splitVals = new double[count];
		if (this.m_smoothPredictions) {
			this.m_internalNodes = new RuleNode[count];
		}

		// trace back to the root
		int i = 0;

		while (temp.parentNode() != null) {
			this.m_splitAtts[i] = temp.parentNode().splitAtt();
			this.m_splitVals[i] = temp.parentNode().splitVal();

			if (temp.parentNode().leftNode() == temp) {
				this.m_relOps[i] = LEFT;
				temp.parentNode().m_right = null;
			} else {
				this.m_relOps[i] = RIGHT;
				temp.parentNode().m_left = null;
			}

			if (this.m_smoothPredictions) {
				this.m_internalNodes[i] = temp.parentNode();
			}

			temp = temp.parentNode();
			i++;
		}

		// now assemble the covered and uncovered instances
		boolean ok;

		for (i = 0; i < this.m_numInstances; i++) {
			ok = true;

			for (int j = 0; j < this.m_relOps.length; j++) {
				if (this.m_relOps[j] == LEFT) {
					if (this.m_instances.instance(i).value(this.m_splitAtts[j]) > this.m_splitVals[j]) {
						this.m_notCovered.add(this.m_instances.instance(i));
						ok = false;
						break;
					}
				} else {
					if (this.m_instances.instance(i).value(this.m_splitAtts[j]) <= this.m_splitVals[j]) {
						this.m_notCovered.add(this.m_instances.instance(i));
						ok = false;
						break;
					}
				}
			}

			if (ok) {
				this.m_numCovered++;
				// m_covered.add(m_instances.instance(i));
			}
		}
	}

	/**
	 * Return a description of the m5 tree or rule
	 * 
	 * @return a description of the m5 tree or rule as a String
	 */
	@Override
	public String toString() {
		if (this.m_useTree) {
			return this.treeToString();
		} else {
			return this.ruleToString();
		}
	}

	/**
	 * Return a description of the m5 tree
	 * 
	 * @return a description of the m5 tree as a String
	 */
	private String treeToString() {
		StringBuffer text = new StringBuffer();

		if (this.m_topOfTree == null) {
			return "Tree/Rule has not been built yet!";
		}

		text.append("M5 " + ((this.m_useUnpruned) ? "unpruned " : "pruned ") + ((this.m_regressionTree) ? "regression " : "model ") + "tree:\n");

		if (this.m_smoothPredictions == true) {
			text.append("(using smoothed linear models)\n");
		}

		text.append(this.m_topOfTree.treeToString(0));
		text.append(this.m_topOfTree.printLeafModels());
		text.append("\nNumber of Rules : " + this.m_topOfTree.numberOfLinearModels());

		return text.toString();
	}

	/**
	 * Return a description of the rule
	 * 
	 * @return a description of the rule as a String
	 */
	private String ruleToString() {
		StringBuffer text = new StringBuffer();

		if (this.m_splitAtts.length > 0) {
			text.append("IF\n");

			for (int i = this.m_splitAtts.length - 1; i >= 0; i--) {
				text.append("\t" + this.m_covered.attribute(this.m_splitAtts[i]).name() + " ");

				if (this.m_relOps[i] == 0) {
					text.append("<= ");
				} else {
					text.append("> ");
				}

				text.append(Utils.doubleToString(this.m_splitVals[i], 1, 3) + "\n");
			}

			text.append("THEN\n");
		}

		if (this.m_ruleModel != null) {
			try {
				text.append(this.m_ruleModel.printNodeLinearModel());
				text.append(" [" + this.m_numCovered/* m_covered.numInstances() */);

				if (this.m_globalAbsDev > 0.0) {
					text.append("/" + Utils.doubleToString((100 * this.m_ruleModel.rootMeanSquaredError() / this.m_globalStdDev), 1, 3) + "%]\n\n");
				} else {
					text.append("]\n\n");
				}
			} catch (Exception e) {
				return "Can't print rule";
			}
		}

		return text.toString();
	}

	/**
	 * Use unpruned tree/rules
	 * 
	 * @param unpruned
	 *            true if unpruned tree/rules are to be generated
	 */
	public void setUnpruned(final boolean unpruned) {
		this.m_useUnpruned = unpruned;
	}

	/**
	 * Get whether unpruned tree/rules are being generated
	 * 
	 * @return true if unpruned tree/rules are to be generated
	 */
	public boolean getUnpruned() {
		return this.m_useUnpruned;
	}

	/**
	 * Use an m5 tree rather than generate rules
	 * 
	 * @param u
	 *            true if m5 tree is to be used
	 */
	public void setUseTree(final boolean u) {
		this.m_useTree = u;
	}

	/**
	 * get whether an m5 tree is being used rather than rules
	 * 
	 * @return true if an m5 tree is being used.
	 */
	public boolean getUseTree() {
		return this.m_useTree;
	}

	/**
	 * Smooth predictions
	 * 
	 * @param s
	 *            true if smoothing is to be used
	 */
	public void setSmoothing(final boolean s) {
		this.m_smoothPredictions = s;
	}

	/**
	 * Get whether or not smoothing has been turned on
	 * 
	 * @return true if smoothing is being used
	 */
	public boolean getSmoothing() {
		return this.m_smoothPredictions;
	}

	/**
	 * Get the instances not covered by this rule
	 * 
	 * @return the instances not covered
	 */
	public Instances notCoveredInstances() {
		return this.m_notCovered;
	}

	/**
	 * Free up memory consumed by the set of instances not covered by this rule.
	 */
	public void freeNotCoveredInstances() {
		this.m_notCovered = null;
	}

	// /**
	// * Get the instances covered by this rule
	// *
	// * @return the instances covered by this rule
	// */
	// public Instances coveredInstances() {
	// return m_covered;
	// }

	/**
	 * Returns the standard deviation value of the supplied attribute index.
	 * 
	 * @param attr
	 *            an attribute index
	 * @param inst
	 *            the instances
	 * @return the standard deviation value
	 */
	protected static final double stdDev(final int attr, final Instances inst) {
		int i, count = 0;
		double sd, va, sum = 0.0, sqrSum = 0.0, value;

		for (i = 0; i <= inst.numInstances() - 1; i++) {
			count++;
			value = inst.instance(i).value(attr);
			sum += value;
			sqrSum += value * value;
		}

		if (count > 1) {
			va = (sqrSum - sum * sum / count) / count;
			va = Math.abs(va);
			sd = Math.sqrt(va);
		} else {
			sd = 0.0;
		}

		return sd;
	}

	/**
	 * Returns the absolute deviation value of the supplied attribute index.
	 * 
	 * @param attr
	 *            an attribute index
	 * @param inst
	 *            the instances
	 * @return the absolute deviation value
	 */
	protected static final double absDev(final int attr, final Instances inst) {
		int i;
		double average = 0.0, absdiff = 0.0, absDev;

		for (i = 0; i <= inst.numInstances() - 1; i++) {
			average += inst.instance(i).value(attr);
		}
		if (inst.numInstances() > 1) {
			average /= inst.numInstances();
			for (i = 0; i <= inst.numInstances() - 1; i++) {
				absdiff += Math.abs(inst.instance(i).value(attr) - average);
			}
			absDev = absdiff / inst.numInstances();
		} else {
			absDev = 0.0;
		}

		return absDev;
	}

	/**
	 * Sets whether instances at each node in an M5 tree should be saved for visualization purposes. Default is to save memory.
	 * 
	 * @param save
	 *            a <code>boolean</code> value
	 */
	protected void setSaveInstances(final boolean save) {
		this.m_saveInstances = save;
	}

	/**
	 * Get the value of regressionTree.
	 * 
	 * @return Value of regressionTree.
	 */
	public boolean getRegressionTree() {

		return this.m_regressionTree;
	}

	/**
	 * Set the value of regressionTree.
	 * 
	 * @param newregressionTree
	 *            Value to assign to regressionTree.
	 */
	public void setRegressionTree(final boolean newregressionTree) {

		this.m_regressionTree = newregressionTree;
	}

	/**
	 * Set the minumum number of instances to allow at a leaf node
	 * 
	 * @param minNum
	 *            the minimum number of instances
	 */
	public void setMinNumInstances(final double minNum) {
		this.m_minNumInstances = minNum;
	}

	/**
	 * Get the minimum number of instances to allow at a leaf node
	 * 
	 * @return a <code>double</code> value
	 */
	public double getMinNumInstances() {
		return this.m_minNumInstances;
	}

	public RuleNode getM5RootNode() {
		return this.m_topOfTree;
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
