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
 *    RuleNode.java
 *    Copyright (C) 2000 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.m5;

import java.util.ArrayList;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Constructs a node for use in an m5 tree or rule
 *
 * @author Mark Hall (mhall@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class RuleNode extends AbstractClassifier {

	/** for serialization */
	static final long serialVersionUID = 1979807611124337144L;

	/**
	 * instances reaching this node
	 */
	private Instances m_instances;

	/**
	 * the class index
	 */
	private int m_classIndex;

	/**
	 * the number of instances reaching this node
	 */
	protected int m_numInstances;

	/**
	 * the number of attributes
	 */
	private int m_numAttributes;

	/**
	 * Node is a leaf
	 */
	private boolean m_isLeaf;

	/**
	 * attribute this node splits on
	 */
	private int m_splitAtt;

	/**
	 * the value of the split attribute
	 */
	private double m_splitValue;

	/**
	 * the linear model at this node
	 */
	private PreConstructedLinearModel m_nodeModel;

	/**
	 * the number of paramters in the chosen model for this node---either the subtree model or the linear model. The constant term is counted as a paramter---this is for pruning purposes
	 */
	public int m_numParameters;

	/**
	 * the mean squared error of the model at this node (either linear or subtree)
	 */
	private double m_rootMeanSquaredError;

	/**
	 * left child node
	 */
	protected RuleNode m_left;

	/**
	 * right child node
	 */
	protected RuleNode m_right;

	/**
	 * the parent of this node
	 */
	private final RuleNode m_parent;

	/**
	 * a node will not be split if it contains less then m_splitNum instances
	 */
	private double m_splitNum = 4;

	/**
	 * a node will not be split if its class standard deviation is less than 5% of the class standard deviation of all the instances
	 */
	private final double m_devFraction = 0.05;
	private final double m_pruningMultiplier = 2;

	/**
	 * the number assigned to the linear model if this node is a leaf. = 0 if this node is not a leaf
	 */
	private int m_leafModelNum;

	/**
	 * a node will not be split if the class deviation of its instances is less than m_devFraction of the deviation of the global class
	 */
	private final double m_globalDeviation;

	/**
	 * the absolute deviation of the global class
	 */
	private final double m_globalAbsDeviation;

	/**
	 * Indices of the attributes to be used in generating a linear model at this node
	 */
	private int[] m_indices;

	/**
	 * Constant used in original m5 smoothing calculation
	 */
	private static final double SMOOTHING_CONSTANT = 15.0;

	/**
	 * Node id.
	 */
	private int m_id;

	/**
	 * Save the instances at each node (for visualizing in the Explorer's treevisualizer.
	 */
	private boolean m_saveInstances = false;

	/**
	 * Make a regression tree instead of a model tree
	 */
	private boolean m_regressionTree;

	/**
	 * Creates a new <code>RuleNode</code> instance.
	 * 
	 * @param globalDev
	 *            the global standard deviation of the class
	 * @param globalAbsDev
	 *            the global absolute deviation of the class
	 * @param parent
	 *            the parent of this node
	 */
	public RuleNode(final double globalDev, final double globalAbsDev, final RuleNode parent) {
		this.m_nodeModel = null;
		this.m_right = null;
		this.m_left = null;
		this.m_parent = parent;
		this.m_globalDeviation = globalDev;
		this.m_globalAbsDeviation = globalAbsDev;
	}

	/**
	 * Build this node (find an attribute and split point)
	 * 
	 * @param data
	 *            the instances on which to build this node
	 * @throws Exception
	 *             if an error occurs
	 */
	@Override
	public void buildClassifier(final Instances data) throws Exception {

		this.m_rootMeanSquaredError = Double.MAX_VALUE;
		// m_instances = new Instances(data);
		this.m_instances = data;
		this.m_classIndex = this.m_instances.classIndex();
		this.m_numInstances = this.m_instances.numInstances();
		this.m_numAttributes = this.m_instances.numAttributes();
		this.m_nodeModel = null;
		this.m_right = null;
		this.m_left = null;

		if ((this.m_numInstances < this.m_splitNum) || (Rule.stdDev(this.m_classIndex, this.m_instances) < (this.m_globalDeviation * this.m_devFraction))) {
			this.m_isLeaf = true;
		} else {
			this.m_isLeaf = false;
		}

		this.split();
	}

	/**
	 * Classify an instance using this node. Recursively calls classifyInstance on child nodes.
	 * 
	 * @param inst
	 *            the instance to classify
	 * @return the prediction for this instance
	 * @throws Exception
	 *             if an error occurs
	 */
	@Override
	public double classifyInstance(final Instance inst) throws Exception {
		if (this.m_isLeaf) {
			if (this.m_nodeModel == null) {
				throw new Exception("Classifier has not been built correctly.");
			}

			return this.m_nodeModel.classifyInstance(inst);
		}

		if (inst.value(this.m_splitAtt) <= this.m_splitValue) {
			return this.m_left.classifyInstance(inst);
		} else {
			return this.m_right.classifyInstance(inst);
		}
	}

	/**
	 * Applies the m5 smoothing procedure to a prediction
	 * 
	 * @param n
	 *            number of instances in selected child of this node
	 * @param pred
	 *            the prediction so far
	 * @param supportPred
	 *            the prediction of the linear model at this node
	 * @return the current prediction smoothed with the prediction of the linear model at this node
	 * @throws Exception
	 *             if an error occurs
	 */
	protected static double smoothingOriginal(final double n, final double pred, final double supportPred) throws Exception {
		double smoothed;

		smoothed = ((n * pred) + (SMOOTHING_CONSTANT * supportPred)) / (n + SMOOTHING_CONSTANT);

		return smoothed;
	}

	/**
	 * Finds an attribute and split point for this node
	 * 
	 * @throws Exception
	 *             if an error occurs
	 */
	public void split() throws Exception {
		int i;
		Instances leftSubset, rightSubset;
		SplitEvaluate bestSplit, currentSplit;
		boolean[] attsBelow;

		if (!this.m_isLeaf) {

			bestSplit = new YongSplitInfo(0, this.m_numInstances - 1, -1);
			currentSplit = new YongSplitInfo(0, this.m_numInstances - 1, -1);

			// find the best attribute to split on
			for (i = 0; i < this.m_numAttributes; i++) {
				if (i != this.m_classIndex) {

					// sort the instances by this attribute
					this.m_instances.sort(i);
					currentSplit.attrSplit(i, this.m_instances);

					if ((Math.abs(currentSplit.maxImpurity() - bestSplit.maxImpurity()) > 1.e-6) && (currentSplit.maxImpurity() > bestSplit.maxImpurity() + 1.e-6)) {
						bestSplit = currentSplit.copy();
					}
				}
			}

			// cant find a good split or split point?
			if (bestSplit.splitAttr() < 0 || bestSplit.position() < 1 || bestSplit.position() > this.m_numInstances - 1) {
				this.m_isLeaf = true;
			} else {
				this.m_splitAtt = bestSplit.splitAttr();
				this.m_splitValue = bestSplit.splitValue();
				leftSubset = new Instances(this.m_instances, this.m_numInstances);
				rightSubset = new Instances(this.m_instances, this.m_numInstances);

				for (i = 0; i < this.m_numInstances; i++) {
					if (this.m_instances.instance(i).value(this.m_splitAtt) <= this.m_splitValue) {
						leftSubset.add(this.m_instances.instance(i));
					} else {
						rightSubset.add(this.m_instances.instance(i));
					}
				}

				leftSubset.compactify();
				rightSubset.compactify();

				// build left and right nodes
				this.m_left = new RuleNode(this.m_globalDeviation, this.m_globalAbsDeviation, this);
				this.m_left.setMinNumInstances(this.m_splitNum);
				this.m_left.setRegressionTree(this.m_regressionTree);
				this.m_left.setSaveInstances(this.m_saveInstances);
				this.m_left.buildClassifier(leftSubset);

				this.m_right = new RuleNode(this.m_globalDeviation, this.m_globalAbsDeviation, this);
				this.m_right.setMinNumInstances(this.m_splitNum);
				this.m_right.setRegressionTree(this.m_regressionTree);
				this.m_right.setSaveInstances(this.m_saveInstances);
				this.m_right.buildClassifier(rightSubset);

				// now find out what attributes are tested in the left and right
				// subtrees and use them to learn a linear model for this node
				if (!this.m_regressionTree) {
					attsBelow = this.attsTestedBelow();
					attsBelow[this.m_classIndex] = true;
					int count = 0, j;

					for (j = 0; j < this.m_numAttributes; j++) {
						if (attsBelow[j]) {
							count++;
						}
					}

					int[] indices = new int[count];

					count = 0;

					for (j = 0; j < this.m_numAttributes; j++) {
						if (attsBelow[j] && (j != this.m_classIndex)) {
							indices[count++] = j;
						}
					}

					indices[count] = this.m_classIndex;
					this.m_indices = indices;
				} else {
					this.m_indices = new int[1];
					this.m_indices[0] = this.m_classIndex;
					this.m_numParameters = 1;
				}
			}
		}

		if (this.m_isLeaf) {
			int[] indices = new int[1];
			indices[0] = this.m_classIndex;
			this.m_indices = indices;
			this.m_numParameters = 1;

			// need to evaluate the model here if want correct stats for unpruned
			// tree
		}
	}

	/**
	 * Build a linear model for this node using those attributes specified in indices.
	 * 
	 * @param indices
	 *            an array of attribute indices to include in the linear model
	 * @throws Exception
	 *             if something goes wrong
	 */
	private void buildLinearModel(final int[] indices) throws Exception {
		// copy the training instances and remove all but the tested
		// attributes
		Instances reducedInst = new Instances(this.m_instances);
		Remove attributeFilter = new Remove();

		attributeFilter.setInvertSelection(true);
		attributeFilter.setAttributeIndicesArray(indices);
		attributeFilter.setInputFormat(reducedInst);

		reducedInst = Filter.useFilter(reducedInst, attributeFilter);

		// build a linear regression for the training data using the
		// tested attributes
		LinearRegression temp = new LinearRegression();
		temp.setDoNotCheckCapabilities(true);
		temp.setMinimal(true);
		temp.buildClassifier(reducedInst);

		double[] lmCoeffs = temp.coefficients();
		double[] coeffs = new double[this.m_instances.numAttributes()];

		for (int i = 0; i < lmCoeffs.length - 1; i++) {
			if (indices[i] != this.m_classIndex) {
				coeffs[indices[i]] = lmCoeffs[i];
			}
		}
		this.m_nodeModel = new PreConstructedLinearModel(coeffs, lmCoeffs[lmCoeffs.length - 1]);
		this.m_nodeModel.buildClassifier(this.m_instances);
	}

	/**
	 * Returns an array containing the indexes of attributes used in tests below this node
	 * 
	 * @return an array of attribute indexes
	 */
	private boolean[] attsTestedBelow() {
		boolean[] attsBelow = new boolean[this.m_numAttributes];
		boolean[] attsBelowLeft = null;
		boolean[] attsBelowRight = null;

		if (this.m_right != null) {
			attsBelowRight = this.m_right.attsTestedBelow();
		}

		if (this.m_left != null) {
			attsBelowLeft = this.m_left.attsTestedBelow();
		}

		for (int i = 0; i < this.m_numAttributes; i++) {
			if (attsBelowLeft != null) {
				attsBelow[i] = (attsBelow[i] || attsBelowLeft[i]);
			}

			if (attsBelowRight != null) {
				attsBelow[i] = (attsBelow[i] || attsBelowRight[i]);
			}
		}

		if (!this.m_isLeaf) {
			attsBelow[this.m_splitAtt] = true;
		}
		return attsBelow;
	}

	/**
	 * Sets the leaves' numbers
	 * 
	 * @param leafCounter
	 *            the number of leaves counted
	 * @return the number of the total leaves under the node
	 */
	public int numLeaves(int leafCounter) {

		if (!this.m_isLeaf) {
			// node
			this.m_leafModelNum = 0;

			if (this.m_left != null) {
				leafCounter = this.m_left.numLeaves(leafCounter);
			}

			if (this.m_right != null) {
				leafCounter = this.m_right.numLeaves(leafCounter);
			}
		} else {
			// leaf
			leafCounter++;
			this.m_leafModelNum = leafCounter;
		}
		return leafCounter;
	}

	/**
	 * print the linear model at this node
	 * 
	 * @return the linear model
	 */
	@Override
	public String toString() {
		return this.printNodeLinearModel();
	}

	/**
	 * print the linear model at this node
	 * 
	 * @return the linear model at this node
	 */
	public String printNodeLinearModel() {
		return this.m_nodeModel.toString();
	}

	/**
	 * print all leaf models
	 * 
	 * @return the leaf models
	 */
	public String printLeafModels() {
		StringBuffer text = new StringBuffer();

		if (this.m_isLeaf) {
			text.append("\nLM num: " + this.m_leafModelNum);
			text.append(this.m_nodeModel.toString());
			text.append("\n");
		} else {
			text.append(this.m_left.printLeafModels());
			text.append(this.m_right.printLeafModels());
		}
		return text.toString();
	}

	/**
	 * Returns a description of this node (debugging purposes)
	 * 
	 * @return a string describing this node
	 */
	public String nodeToString() {
		StringBuffer text = new StringBuffer();

		text.append("Node:\n\tnum inst: " + this.m_numInstances);

		if (this.m_isLeaf) {
			text.append("\n\tleaf");
		} else {
			text.append("\tnode");
		}

		text.append("\n\tSplit att: " + this.m_instances.attribute(this.m_splitAtt).name());
		text.append("\n\tSplit val: " + Utils.doubleToString(this.m_splitValue, 1, 3));
		text.append("\n\tLM num: " + this.m_leafModelNum);
		text.append("\n\tLinear model\n" + this.m_nodeModel.toString());
		text.append("\n\n");

		if (this.m_left != null) {
			text.append(this.m_left.nodeToString());
		}

		if (this.m_right != null) {
			text.append(this.m_right.nodeToString());
		}

		return text.toString();
	}

	/**
	 * Recursively builds a textual description of the tree
	 * 
	 * @param level
	 *            the level of this node
	 * @return string describing the tree
	 */
	public String treeToString(final int level) {
		int i;
		StringBuffer text = new StringBuffer();

		if (!this.m_isLeaf) {
			text.append("\n");

			for (i = 1; i <= level; i++) {
				text.append("|   ");
			}

			if (this.m_instances.attribute(this.m_splitAtt).name().charAt(0) != '[') {
				text.append(this.m_instances.attribute(this.m_splitAtt).name() + " <= " + Utils.doubleToString(this.m_splitValue, 1, 3) + " : ");
			} else {
				text.append(this.m_instances.attribute(this.m_splitAtt).name() + " false : ");
			}

			if (this.m_left != null) {
				text.append(this.m_left.treeToString(level + 1));
			} else {
				text.append("NULL\n");
			}

			for (i = 1; i <= level; i++) {
				text.append("|   ");
			}

			if (this.m_instances.attribute(this.m_splitAtt).name().charAt(0) != '[') {
				text.append(this.m_instances.attribute(this.m_splitAtt).name() + " >  " + Utils.doubleToString(this.m_splitValue, 1, 3) + " : ");
			} else {
				text.append(this.m_instances.attribute(this.m_splitAtt).name() + " true : ");
			}

			if (this.m_right != null) {
				text.append(this.m_right.treeToString(level + 1));
			} else {
				text.append("NULL\n");
			}
		} else {
			text.append("LM" + this.m_leafModelNum);

			if (this.m_globalDeviation > 0.0) {
				text.append(" (" + this.m_numInstances + "/" + Utils.doubleToString((100.0 * this.m_rootMeanSquaredError / this.m_globalDeviation), 1, 3) + "%)\n");
			} else {
				text.append(" (" + this.m_numInstances + ")\n");
			}
		}
		return text.toString();
	}

	/**
	 * Traverses the tree and installs linear models at each node. This method must be called if pruning is not to be performed.
	 * 
	 * @throws Exception
	 *             if an error occurs
	 */
	public void installLinearModels() throws Exception {
		Evaluation nodeModelEval;
		if (this.m_isLeaf) {
			this.buildLinearModel(this.m_indices);
		} else {
			if (this.m_left != null) {
				this.m_left.installLinearModels();
			}

			if (this.m_right != null) {
				this.m_right.installLinearModels();
			}
			this.buildLinearModel(this.m_indices);
		}
		nodeModelEval = new Evaluation(this.m_instances);
		nodeModelEval.evaluateModel(this.m_nodeModel, this.m_instances);
		this.m_rootMeanSquaredError = nodeModelEval.rootMeanSquaredError();
		// save space
		if (!this.m_saveInstances) {
			this.m_instances = new Instances(this.m_instances, 0);
		}
	}

	/**
	 * 
	 * @throws Exception
	 */
	public void installSmoothedModels() throws Exception {

		if (this.m_isLeaf) {
			double[] coefficients = new double[this.m_numAttributes];
			double intercept;
			double[] coeffsUsedByLinearModel = this.m_nodeModel.coefficients();
			RuleNode current = this;

			// prime array with leaf node coefficients
			for (int i = 0; i < coeffsUsedByLinearModel.length; i++) {
				if (i != this.m_classIndex) {
					coefficients[i] = coeffsUsedByLinearModel[i];
				}
			}
			// intercept
			intercept = this.m_nodeModel.intercept();

			do {
				if (current.m_parent != null) {
					double n = current.m_numInstances;
					// contribution of the model below
					for (int i = 0; i < coefficients.length; i++) {
						coefficients[i] = ((coefficients[i] * n) / (n + SMOOTHING_CONSTANT));
					}
					intercept = ((intercept * n) / (n + SMOOTHING_CONSTANT));

					// contribution of this model
					coeffsUsedByLinearModel = current.m_parent.getModel().coefficients();
					for (int i = 0; i < coeffsUsedByLinearModel.length; i++) {
						if (i != this.m_classIndex) {
							// smooth in these coefficients (at this node)
							coefficients[i] += ((SMOOTHING_CONSTANT * coeffsUsedByLinearModel[i]) / (n + SMOOTHING_CONSTANT));
						}
					}
					// smooth in the intercept
					intercept += ((SMOOTHING_CONSTANT * current.m_parent.getModel().intercept()) / (n + SMOOTHING_CONSTANT));
					current = current.m_parent;
				}
			} while (current.m_parent != null);
			this.m_nodeModel = new PreConstructedLinearModel(coefficients, intercept);
			this.m_nodeModel.buildClassifier(this.m_instances);
		}
		if (this.m_left != null) {
			this.m_left.installSmoothedModels();
		}
		if (this.m_right != null) {
			this.m_right.installSmoothedModels();
		}
	}

	/**
	 * Recursively prune the tree
	 * 
	 * @throws Exception
	 *             if an error occurs
	 */
	public void prune() throws Exception {
		Evaluation nodeModelEval = null;

		if (this.m_isLeaf) {
			this.buildLinearModel(this.m_indices);
			nodeModelEval = new Evaluation(this.m_instances);

			// count the constant term as a paramter for a leaf
			// Evaluate the model
			nodeModelEval.evaluateModel(this.m_nodeModel, this.m_instances);

			this.m_rootMeanSquaredError = nodeModelEval.rootMeanSquaredError();
		} else {

			// Prune the left and right subtrees
			if (this.m_left != null) {
				this.m_left.prune();
			}

			if (this.m_right != null) {
				this.m_right.prune();
			}

			this.buildLinearModel(this.m_indices);
			nodeModelEval = new Evaluation(this.m_instances);

			double rmsModel;
			double adjustedErrorModel;

			nodeModelEval.evaluateModel(this.m_nodeModel, this.m_instances);

			rmsModel = nodeModelEval.rootMeanSquaredError();
			adjustedErrorModel = rmsModel * this.pruningFactor(this.m_numInstances, this.m_nodeModel.numParameters() + 1);

			// Evaluate this node (ie its left and right subtrees)
			Evaluation nodeEval = new Evaluation(this.m_instances);
			double rmsSubTree;
			double adjustedErrorNode;
			int l_params = 0, r_params = 0;

			nodeEval.evaluateModel(this, this.m_instances);

			rmsSubTree = nodeEval.rootMeanSquaredError();

			if (this.m_left != null) {
				l_params = this.m_left.numParameters();
			}

			if (this.m_right != null) {
				r_params = this.m_right.numParameters();
			}

			adjustedErrorNode = rmsSubTree * this.pruningFactor(this.m_numInstances, (l_params + r_params + 1));

			if ((adjustedErrorModel <= adjustedErrorNode) || (adjustedErrorModel < (this.m_globalDeviation * 0.00001))) {

				// Choose linear model for this node rather than subtree model
				this.m_isLeaf = true;
				this.m_right = null;
				this.m_left = null;
				this.m_numParameters = this.m_nodeModel.numParameters() + 1;
				this.m_rootMeanSquaredError = rmsModel;
			} else {
				this.m_numParameters = (l_params + r_params + 1);
				this.m_rootMeanSquaredError = rmsSubTree;
			}
		}
		// save space
		if (!this.m_saveInstances) {
			this.m_instances = new Instances(this.m_instances, 0);
		}
	}

	/**
	 * Compute the pruning factor
	 * 
	 * @param num_instances
	 *            number of instances
	 * @param num_params
	 *            number of parameters in the model
	 * @return the pruning factor
	 */
	private double pruningFactor(final int num_instances, final int num_params) {
		if (num_instances <= num_params) {
			return 10.0; // Caution says Yong in his code
		}

		return ((num_instances + this.m_pruningMultiplier * num_params) / (num_instances - num_params));
	}

	/**
	 * Find the leaf with greatest coverage
	 * 
	 * @param maxCoverage
	 *            the greatest coverage found so far
	 * @param bestLeaf
	 *            the leaf with the greatest coverage
	 */
	public void findBestLeaf(final double[] maxCoverage, final RuleNode[] bestLeaf) {
		if (!this.m_isLeaf) {
			if (this.m_left != null) {
				this.m_left.findBestLeaf(maxCoverage, bestLeaf);
			}

			if (this.m_right != null) {
				this.m_right.findBestLeaf(maxCoverage, bestLeaf);
			}
		} else {
			if (this.m_numInstances > maxCoverage[0]) {
				maxCoverage[0] = this.m_numInstances;
				bestLeaf[0] = this;
			}
		}
	}

	/**
	 * Return a list containing all the leaves in the tree
	 * 
	 * @param v
	 *            a single element array containing a vector of leaves
	 */
	public void returnLeaves(final ArrayList<RuleNode>[] v) {
		if (this.m_isLeaf) {
			v[0].add(this);
		} else {
			if (this.m_left != null) {
				this.m_left.returnLeaves(v);
			}

			if (this.m_right != null) {
				this.m_right.returnLeaves(v);
			}
		}
	}

	/**
	 * Get the parent of this node
	 * 
	 * @return the parent of this node
	 */
	public RuleNode parentNode() {
		return this.m_parent;
	}

	/**
	 * Get the left child of this node
	 * 
	 * @return the left child of this node
	 */
	public RuleNode leftNode() {
		return this.m_left;
	}

	/**
	 * Get the right child of this node
	 * 
	 * @return the right child of this node
	 */
	public RuleNode rightNode() {
		return this.m_right;
	}

	/**
	 * Get the index of the splitting attribute for this node
	 * 
	 * @return the index of the splitting attribute
	 */
	public int splitAtt() {
		return this.m_splitAtt;
	}

	/**
	 * Get the split point for this node
	 * 
	 * @return the split point for this node
	 */
	public double splitVal() {
		return this.m_splitValue;
	}

	/**
	 * Get the number of linear models in the tree
	 * 
	 * @return the number of linear models
	 */
	public int numberOfLinearModels() {
		if (this.m_isLeaf) {
			return 1;
		} else {
			return this.m_left.numberOfLinearModels() + this.m_right.numberOfLinearModels();
		}
	}

	/**
	 * Return true if this node is a leaf
	 * 
	 * @return true if this node is a leaf
	 */
	public boolean isLeaf() {
		return this.m_isLeaf;
	}

	/**
	 * Get the root mean squared error at this node
	 * 
	 * @return the root mean squared error
	 */
	protected double rootMeanSquaredError() {
		return this.m_rootMeanSquaredError;
	}

	/**
	 * Get the linear model at this node
	 * 
	 * @return the linear model at this node
	 */
	public PreConstructedLinearModel getModel() {
		return this.m_nodeModel;
	}

	/**
	 * Return the number of instances that reach this node.
	 * 
	 * @return the number of instances at this node.
	 */
	public int getNumInstances() {
		return this.m_numInstances;
	}

	/**
	 * Get the number of parameters in the model at this node
	 * 
	 * @return the number of parameters in the model at this node
	 */
	private int numParameters() {
		return this.m_numParameters;
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
	 * Set the minumum number of instances to allow at a leaf node
	 * 
	 * @param minNum
	 *            the minimum number of instances
	 */
	public void setMinNumInstances(final double minNum) {
		this.m_splitNum = minNum;
	}

	/**
	 * Get the minimum number of instances to allow at a leaf node
	 * 
	 * @return a <code>double</code> value
	 */
	public double getMinNumInstances() {
		return this.m_splitNum;
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
	 * Print all the linear models at the learf (debugging purposes)
	 */
	public void printAllModels() {
		if (this.m_isLeaf) {
			System.out.println(this.m_nodeModel.toString());
		} else {
			System.out.println(this.m_nodeModel.toString());
			this.m_left.printAllModels();
			this.m_right.printAllModels();
		}
	}

	/**
	 * Assigns a unique identifier to each node in the tree
	 * 
	 * @param lastID
	 *            last id number used
	 * @return ID after processing child nodes
	 */
	protected int assignIDs(final int lastID) {
		int currLastID = lastID + 1;
		this.m_id = currLastID;

		if (this.m_left != null) {
			currLastID = this.m_left.assignIDs(currLastID);
		}

		if (this.m_right != null) {
			currLastID = this.m_right.assignIDs(currLastID);
		}
		return currLastID;
	}

	/**
	 * Assign a unique identifier to each node in the tree and then calls graphTree
	 * 
	 * @param text
	 *            a <code>StringBuffer</code> value
	 */
	public void graph(final StringBuffer text) {
		this.assignIDs(-1);
		this.graphTree(text);
	}

	/**
	 * Return a dotty style string describing the tree
	 * 
	 * @param text
	 *            a <code>StringBuffer</code> value
	 */
	protected void graphTree(final StringBuffer text) {
		text.append("N" + this.m_id + (this.m_isLeaf ? " [label=\"LM " + this.m_leafModelNum : " [label=\"" + Utils.backQuoteChars(this.m_instances.attribute(this.m_splitAtt).name()))
				+ (this.m_isLeaf
						? " (" + ((this.m_globalDeviation > 0.0) ? this.m_numInstances + "/" + Utils.doubleToString((100.0 * this.m_rootMeanSquaredError / this.m_globalDeviation), 1, 3) + "%)" : this.m_numInstances + ")")
								+ "\" shape=box style=filled "
						: "\"")
				+ (this.m_saveInstances ? "data=\n" + this.m_instances + "\n,\n" : "") + "]\n");

		if (this.m_left != null) {
			text.append("N" + this.m_id + "->" + "N" + this.m_left.m_id + " [label=\"<=" + Utils.doubleToString(this.m_splitValue, 1, 3) + "\"]\n");
			this.m_left.graphTree(text);
		}

		if (this.m_right != null) {
			text.append("N" + this.m_id + "->" + "N" + this.m_right.m_id + " [label=\">" + Utils.doubleToString(this.m_splitValue, 1, 3) + "\"]\n");
			this.m_right.graphTree(text);
		}
	}

	/**
	 * Set whether to save instances for visualization purposes. Default is to save memory.
	 * 
	 * @param save
	 *            a <code>boolean</code> value
	 */
	protected void setSaveInstances(final boolean save) {
		this.m_saveInstances = save;
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
