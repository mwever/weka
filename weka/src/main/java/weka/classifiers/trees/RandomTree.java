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
 *    RandomTree.java
 *    Copyright (C) 2001-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees;

import java.io.Serializable;
import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.ContingencyTables;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.PartitionGenerator;
import weka.core.Randomizable;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.gui.ProgrammaticProperty;

/**
 * <!-- globalinfo-start --> Class for constructing a tree that considers K randomly chosen
 * attributes at each node. Performs no pruning. Also has an option to allow estimation of class
 * probabilities (or target mean in the regression case) based on a hold-out set (backfitting). <br>
 * <br>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p>
 *
 * <pre>
 * -K &lt;number of attributes&gt;
 *  Number of attributes to randomly investigate. (default 0)
 *  (&lt;1 = int(log_2(#predictors)+1)).
 * </pre>
 *
 * <pre>
 * -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default 1)
 * </pre>
 *
 * <pre>
 * -V &lt;minimum variance for split&gt;
 *  Set minimum numeric class variance proportion
 *  of train variance for split (default 1e-3).
 * </pre>
 *
 * <pre>
 * -S &lt;num&gt;
 *  Seed for random number generator.
 *  (default 1)
 * </pre>
 *
 * <pre>
 * -depth &lt;num&gt;
 *  The maximum depth of the tree, 0 for unlimited.
 *  (default 0)
 * </pre>
 *
 * <pre>
 * -N &lt;num&gt;
 *  Number of folds for backfitting (default 0, no backfitting).
 * </pre>
 *
 * <pre>
 * -U
 *  Allow unclassified instances.
 * </pre>
 *
 * <pre>
 * -B
 *  Break ties randomly when several attributes look equally good.
 * </pre>
 *
 * <pre>
 * -output-debug-info
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 *
 * <pre>
 * -do-not-check-capabilities
 *  If set, classifier capabilities are not checked before classifier is built
 *  (use with caution).
 * </pre>
 *
 * <pre>
 * -num-decimal-places
 *  The number of decimal places for the output of numbers in the model (default 2).
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class RandomTree extends AbstractClassifier implements OptionHandler, WeightedInstancesHandler, Randomizable, Drawable, PartitionGenerator {

	/** for serialization */
	private static final long serialVersionUID = -9051119597407396024L;

	/** The Tree object */
	protected Tree m_Tree = null;

	/** The header information. */
	protected Instances m_Info = null;

	/** Minimum number of instances for leaf. */
	protected double m_MinNum = 1.0;

	/** The number of attributes considered for a split. */
	protected int m_KValue = 0;

	/** The random seed to use. */
	protected int m_randomSeed = 1;

	/** The maximum depth of the tree (0 = unlimited) */
	protected int m_MaxDepth = 0;

	/** Determines how much data is used for backfitting */
	protected int m_NumFolds = 0;

	/** Whether unclassified instances are allowed */
	protected boolean m_AllowUnclassifiedInstances = false;

	/** Whether to break ties randomly. */
	protected boolean m_BreakTiesRandomly = false;

	/** a ZeroR model in case no model can be built from the data */
	protected Classifier m_zeroR;

	/**
	 * The minimum proportion of the total variance (over all the data) required for split.
	 */
	protected double m_MinVarianceProp = 1e-3;

	/** Whether to store the impurity decrease/gain sum */
	protected boolean m_computeImpurityDecreases;

	/**
	 * Indexed by attribute, each two element array contains impurity decrease/gain sum in first element
	 * and count in the second
	 */
	protected double[][] m_impurityDecreasees;

	/**
	 * Returns a string describing classifier
	 *
	 * @return a description suitable for displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return "Class for constructing a tree that considers K randomly " + " chosen attributes at each node. Performs no pruning. Also has" + " an option to allow estimation of class probabilities (or target mean "
				+ "in the regression case) based on a hold-out set (backfitting).";
	}

	/**
	 * Get the array of impurity decrease/gain sums
	 *
	 * @return the array of impurity decrease/gain sums
	 */
	public double[][] getImpurityDecreases() {
		return this.m_impurityDecreasees;
	}

	/**
	 * Set whether to compute/store impurity decreases for variable importance in RandomForest
	 *
	 * @param computeImpurityDecreases
	 *          true to compute and store impurity decrease values for splitting attributes
	 */
	@ProgrammaticProperty
	public void setComputeImpurityDecreases(final boolean computeImpurityDecreases) {
		this.m_computeImpurityDecreases = computeImpurityDecreases;
	}

	/**
	 * Get whether to compute/store impurity decreases for variable importance in RandomForest
	 *
	 * @return true to compute and store impurity decrease values for splitting attributes
	 */
	public boolean getComputeImpurityDecreases() {
		return this.m_computeImpurityDecreases;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String minNumTipText() {
		return "The minimum total weight of the instances in a leaf.";
	}

	/**
	 * Get the value of MinNum.
	 *
	 * @return Value of MinNum.
	 */
	public double getMinNum() {

		return this.m_MinNum;
	}

	/**
	 * Set the value of MinNum.
	 *
	 * @param newMinNum
	 *          Value to assign to MinNum.
	 */
	public void setMinNum(final double newMinNum) {

		this.m_MinNum = newMinNum;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String minVariancePropTipText() {
		return "The minimum proportion of the variance on all the data " + "that needs to be present at a node in order for splitting to " + "be performed in regression trees.";
	}

	/**
	 * Get the value of MinVarianceProp.
	 *
	 * @return Value of MinVarianceProp.
	 */
	public double getMinVarianceProp() {

		return this.m_MinVarianceProp;
	}

	/**
	 * Set the value of MinVarianceProp.
	 *
	 * @param newMinVarianceProp
	 *          Value to assign to MinVarianceProp.
	 */
	public void setMinVarianceProp(final double newMinVarianceProp) {

		this.m_MinVarianceProp = newMinVarianceProp;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String KValueTipText() {
		return "Sets the number of randomly chosen attributes. If 0, int(log_2(#predictors) + 1) is used.";
	}

	/**
	 * Get the value of K.
	 *
	 * @return Value of K.
	 */
	public int getKValue() {

		return this.m_KValue;
	}

	/**
	 * Set the value of K.
	 *
	 * @param k
	 *          Value to assign to K.
	 */
	public void setKValue(final int k) {

		this.m_KValue = k;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String seedTipText() {
		return "The random number seed used for selecting attributes.";
	}

	/**
	 * Set the seed for random number generation.
	 *
	 * @param seed
	 *          the seed
	 */
	@Override
	public void setSeed(final int seed) {

		this.m_randomSeed = seed;
	}

	/**
	 * Gets the seed for the random number generations
	 *
	 * @return the seed for the random number generation
	 */
	@Override
	public int getSeed() {

		return this.m_randomSeed;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String maxDepthTipText() {
		return "The maximum depth of the tree, 0 for unlimited.";
	}

	/**
	 * Get the maximum depth of trh tree, 0 for unlimited.
	 *
	 * @return the maximum depth.
	 */
	public int getMaxDepth() {
		return this.m_MaxDepth;
	}

	/**
	 * Set the maximum depth of the tree, 0 for unlimited.
	 *
	 * @param value
	 *          the maximum depth.
	 */
	public void setMaxDepth(final int value) {
		this.m_MaxDepth = value;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String numFoldsTipText() {
		return "Determines the amount of data used for backfitting. One fold is used for " + "backfitting, the rest for growing the tree. (Default: 0, no backfitting)";
	}

	/**
	 * Get the value of NumFolds.
	 *
	 * @return Value of NumFolds.
	 */
	public int getNumFolds() {

		return this.m_NumFolds;
	}

	/**
	 * Set the value of NumFolds.
	 *
	 * @param newNumFolds
	 *          Value to assign to NumFolds.
	 */
	public void setNumFolds(final int newNumFolds) {

		this.m_NumFolds = newNumFolds;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String allowUnclassifiedInstancesTipText() {
		return "Whether to allow unclassified instances.";
	}

	/**
	 * Gets whether tree is allowed to abstain from making a prediction.
	 *
	 * @return true if tree is allowed to abstain from making a prediction.
	 */
	public boolean getAllowUnclassifiedInstances() {

		return this.m_AllowUnclassifiedInstances;
	}

	/**
	 * Set the value of AllowUnclassifiedInstances.
	 *
	 * @param newAllowUnclassifiedInstances
	 *          true if tree is allowed to abstain from making a prediction
	 */
	public void setAllowUnclassifiedInstances(final boolean newAllowUnclassifiedInstances) {

		this.m_AllowUnclassifiedInstances = newAllowUnclassifiedInstances;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String breakTiesRandomlyTipText() {
		return "Break ties randomly when several attributes look equally good.";
	}

	/**
	 * Get whether to break ties randomly.
	 *
	 * @return true if ties are to be broken randomly.
	 */
	public boolean getBreakTiesRandomly() {

		return this.m_BreakTiesRandomly;
	}

	/**
	 * Set whether to break ties randomly.
	 *
	 * @param newBreakTiesRandomly
	 *          true if ties are to be broken randomly
	 */
	public void setBreakTiesRandomly(final boolean newBreakTiesRandomly) {

		this.m_BreakTiesRandomly = newBreakTiesRandomly;
	}

	/**
	 * Lists the command-line options for this classifier.
	 *
	 * @return an enumeration over all possible options
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<>();

		newVector.addElement(new Option("\tNumber of attributes to randomly investigate.\t(default 0)\n" + "\t(<1 = int(log_2(#predictors)+1)).", "K", 1, "-K <number of attributes>"));

		newVector.addElement(new Option("\tSet minimum number of instances per leaf.\n\t(default 1)", "M", 1, "-M <minimum number of instances>"));

		newVector.addElement(new Option("\tSet minimum numeric class variance proportion\n" + "\tof train variance for split (default 1e-3).", "V", 1, "-V <minimum variance for split>"));

		newVector.addElement(new Option("\tSeed for random number generator.\n" + "\t(default 1)", "S", 1, "-S <num>"));

		newVector.addElement(new Option("\tThe maximum depth of the tree, 0 for unlimited.\n" + "\t(default 0)", "depth", 1, "-depth <num>"));

		newVector.addElement(new Option("\tNumber of folds for backfitting " + "(default 0, no backfitting).", "N", 1, "-N <num>"));
		newVector.addElement(new Option("\tAllow unclassified instances.", "U", 0, "-U"));
		newVector.addElement(new Option("\t" + this.breakTiesRandomlyTipText(), "B", 0, "-B"));
		newVector.addAll(Collections.list(super.listOptions()));

		return newVector.elements();
	}

	/**
	 * Gets options from this classifier.
	 *
	 * @return the options for the current setup
	 */
	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<>();

		result.add("-K");
		result.add("" + this.getKValue());

		result.add("-M");
		result.add("" + this.getMinNum());

		result.add("-V");
		result.add("" + this.getMinVarianceProp());

		result.add("-S");
		result.add("" + this.getSeed());

		if (this.getMaxDepth() > 0) {
			result.add("-depth");
			result.add("" + this.getMaxDepth());
		}

		if (this.getNumFolds() > 0) {
			result.add("-N");
			result.add("" + this.getNumFolds());
		}

		if (this.getAllowUnclassifiedInstances()) {
			result.add("-U");
		}

		if (this.getBreakTiesRandomly()) {
			result.add("-B");
		}

		Collections.addAll(result, super.getOptions());

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 *
	 * <!-- options-start --> Valid options are:
	 * <p>
	 *
	 * <pre>
	 * -K &lt;number of attributes&gt;
	 *  Number of attributes to randomly investigate. (default 0)
	 *  (&lt;1 = int(log_2(#predictors)+1)).
	 * </pre>
	 *
	 * <pre>
	 * -M &lt;minimum number of instances&gt;
	 *  Set minimum number of instances per leaf.
	 *  (default 1)
	 * </pre>
	 *
	 * <pre>
	 * -V &lt;minimum variance for split&gt;
	 *  Set minimum numeric class variance proportion
	 *  of train variance for split (default 1e-3).
	 * </pre>
	 *
	 * <pre>
	 * -S &lt;num&gt;
	 *  Seed for random number generator.
	 *  (default 1)
	 * </pre>
	 *
	 * <pre>
	 * -depth &lt;num&gt;
	 *  The maximum depth of the tree, 0 for unlimited.
	 *  (default 0)
	 * </pre>
	 *
	 * <pre>
	 * -N &lt;num&gt;
	 *  Number of folds for backfitting (default 0, no backfitting).
	 * </pre>
	 *
	 * <pre>
	 * -U
	 *  Allow unclassified instances.
	 * </pre>
	 *
	 * <pre>
	 * -B
	 *  Break ties randomly when several attributes look equally good.
	 * </pre>
	 *
	 * <pre>
	 * -output-debug-info
	 *  If set, classifier is run in debug mode and
	 *  may output additional info to the console
	 * </pre>
	 *
	 * <pre>
	 * -do-not-check-capabilities
	 *  If set, classifier capabilities are not checked before classifier is built
	 *  (use with caution).
	 * </pre>
	 *
	 * <pre>
	 * -num-decimal-places
	 *  The number of decimal places for the output of numbers in the model (default 2).
	 * </pre>
	 *
	 * <!-- options-end -->
	 *
	 * @param options
	 *          the list of options as an array of strings
	 * @throws Exception
	 *           if an option is not supported
	 */
	@Override
	public void setOptions(final String[] options) throws Exception {
		String tmpStr;

		tmpStr = Utils.getOption('K', options);
		if (tmpStr.length() != 0) {
			this.m_KValue = Integer.parseInt(tmpStr);
		} else {
			this.m_KValue = 0;
		}

		tmpStr = Utils.getOption('M', options);
		if (tmpStr.length() != 0) {
			this.m_MinNum = Double.parseDouble(tmpStr);
		} else {
			this.m_MinNum = 1;
		}

		String minVarString = Utils.getOption('V', options);
		if (minVarString.length() != 0) {
			this.m_MinVarianceProp = Double.parseDouble(minVarString);
		} else {
			this.m_MinVarianceProp = 1e-3;
		}

		tmpStr = Utils.getOption('S', options);
		if (tmpStr.length() != 0) {
			this.setSeed(Integer.parseInt(tmpStr));
		} else {
			this.setSeed(1);
		}

		tmpStr = Utils.getOption("depth", options);
		if (tmpStr.length() != 0) {
			this.setMaxDepth(Integer.parseInt(tmpStr));
		} else {
			this.setMaxDepth(0);
		}
		String numFoldsString = Utils.getOption('N', options);
		if (numFoldsString.length() != 0) {
			this.m_NumFolds = Integer.parseInt(numFoldsString);
		} else {
			this.m_NumFolds = 0;
		}

		this.setAllowUnclassifiedInstances(Utils.getFlag('U', options));

		this.setBreakTiesRandomly(Utils.getFlag('B', options));

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		return result;
	}

	/**
	 * Builds classifier.
	 *
	 * @param data
	 *          the data to train with
	 * @throws Exception
	 *           if something goes wrong or the data doesn't fit
	 */
	@Override
	public void buildClassifier(Instances data) throws Exception {

		if (this.m_computeImpurityDecreases) {
			this.m_impurityDecreasees = new double[data.numAttributes()][2];
		}

		// Make sure K value is in range
		if (this.m_KValue > data.numAttributes() - 1) {
			this.m_KValue = data.numAttributes() - 1;
		}
		if (this.m_KValue < 1) {
			this.m_KValue = (int) Utils.log2(data.numAttributes() - 1) + 1;
		}

		// can classifier handle the data?
		this.getCapabilities().testWithFail(data);

		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

		// only class? -> build ZeroR model
		if (data.numAttributes() == 1) {
			System.err.println("Cannot build model (only class attribute present in data!), " + "using ZeroR model instead!");
			this.m_zeroR = new weka.classifiers.rules.ZeroR();
			this.m_zeroR.buildClassifier(data);
			return;
		} else {
			this.m_zeroR = null;
		}

		// Figure out appropriate datasets
		Instances train = null;
		Instances backfit = null;
		Random rand = data.getRandomNumberGenerator(this.m_randomSeed);
		if (this.m_NumFolds <= 0) {
			train = data;
		} else {
			data.randomize(rand);
			data.stratify(this.m_NumFolds);
			train = data.trainCV(this.m_NumFolds, 1, rand);
			backfit = data.testCV(this.m_NumFolds, 1);
		}

		// Create the attribute indices window
		int[] attIndicesWindow = new int[data.numAttributes() - 1];
		int j = 0;
		for (int i = 0; i < attIndicesWindow.length; i++) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			if (j == data.classIndex()) {
				j++; // do not include the class
			}
			attIndicesWindow[i] = j++;
		}

		double totalWeight = 0;
		double totalSumSquared = 0;

		// Compute initial class counts
		double[] classProbs = new double[train.numClasses()];
		for (int i = 0; i < train.numInstances(); i++) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			Instance inst = train.instance(i);
			if (data.classAttribute().isNominal()) {
				classProbs[(int) inst.classValue()] += inst.weight();
				totalWeight += inst.weight();
			} else {
				classProbs[0] += inst.classValue() * inst.weight();
				totalSumSquared += inst.classValue() * inst.classValue() * inst.weight();
				totalWeight += inst.weight();
			}
		}

		double trainVariance = 0;
		if (data.classAttribute().isNumeric()) {
			trainVariance = RandomTree.singleVariance(classProbs[0], totalSumSquared, totalWeight) / totalWeight;
			classProbs[0] /= totalWeight;
		}

		// Build tree
		this.m_Tree = new Tree();
		this.m_Info = new Instances(data, 0);
		this.m_Tree.buildTree(train, classProbs, attIndicesWindow, totalWeight, rand, 0, this.m_MinVarianceProp * trainVariance);

		// Backfit if required
		if (backfit != null) {
			this.m_Tree.backfitData(backfit);
		}
	}

	/**
	 * Computes class distribution of an instance using the tree.
	 *
	 * @param instance
	 *          the instance to compute the distribution for
	 * @return the computed class probabilities
	 * @throws Exception
	 *           if computation fails
	 */
	@Override
	public double[] distributionForInstance(final Instance instance) throws Exception {

		if (this.m_zeroR != null) {
			return this.m_zeroR.distributionForInstance(instance);
		} else {
			return this.m_Tree.distributionForInstance(instance);
		}
	}

	/**
	 * Outputs the decision tree.
	 *
	 * @return a string representation of the classifier
	 */
	@Override
	public String toString() {

		// only ZeroR model?
		if (this.m_zeroR != null) {
			StringBuffer buf = new StringBuffer();
			buf.append(this.getClass().getName().replaceAll(".*\\.", "") + "\n");
			buf.append(this.getClass().getName().replaceAll(".*\\.", "").replaceAll(".", "=") + "\n\n");
			buf.append("Warning: No model could be built, hence ZeroR model is used:\n\n");
			buf.append(this.m_zeroR.toString());
			return buf.toString();
		}

		if (this.m_Tree == null) {
			return "RandomTree: no model has been built yet.";
		} else {
			return "\nRandomTree\n==========\n" + this.m_Tree.toString(0) + "\n" + "\nSize of the tree : " + this.m_Tree.numNodes() + (this.getMaxDepth() > 0 ? ("\nMax depth of tree: " + this.getMaxDepth()) : (""));
		}
	}

	/**
	 * Returns graph describing the tree.
	 *
	 * @return the graph describing the tree
	 * @throws Exception
	 *           if graph can't be computed
	 */
	@Override
	public String graph() throws Exception {

		if (this.m_Tree == null) {
			throw new Exception("RandomTree: No model built yet.");
		}
		StringBuffer resultBuff = new StringBuffer();
		this.m_Tree.toGraph(resultBuff, 0, null);
		String result = "digraph RandomTree {\n" + "edge [style=bold]\n" + resultBuff.toString() + "\n}\n";
		return result;
	}

	/**
	 * Returns the type of graph this classifier represents.
	 *
	 * @return Drawable.TREE
	 */
	@Override
	public int graphType() {
		return Drawable.TREE;
	}

	/**
	 * Builds the classifier to generate a partition.
	 */
	@Override
	public void generatePartition(final Instances data) throws Exception {

		this.buildClassifier(data);
	}

	/**
	 * Computes array that indicates node membership. Array locations are allocated based on
	 * breadth-first exploration of the tree.
	 */
	@Override
	public double[] getMembershipValues(final Instance instance) throws Exception {

		if (this.m_zeroR != null) {
			double[] m = new double[1];
			m[0] = instance.weight();
			return m;
		} else {

			// Set up array for membership values
			double[] a = new double[this.numElements()];

			// Initialize queues
			Queue<Double> queueOfWeights = new LinkedList<>();
			Queue<Tree> queueOfNodes = new LinkedList<>();
			queueOfWeights.add(instance.weight());
			queueOfNodes.add(this.m_Tree);
			int index = 0;

			// While the queue is not empty
			while (!queueOfNodes.isEmpty()) {

				a[index++] = queueOfWeights.poll();
				Tree node = queueOfNodes.poll();

				// Is node a leaf?
				if (node.m_Attribute <= -1) {
					continue;
				}

				// Compute weight distribution
				double[] weights = new double[node.m_Successors.length];
				if (instance.isMissing(node.m_Attribute)) {
					System.arraycopy(node.m_Prop, 0, weights, 0, node.m_Prop.length);
				} else if (this.m_Info.attribute(node.m_Attribute).isNominal()) {
					weights[(int) instance.value(node.m_Attribute)] = 1.0;
				} else {
					if (instance.value(node.m_Attribute) < node.m_SplitPoint) {
						weights[0] = 1.0;
					} else {
						weights[1] = 1.0;
					}
				}
				for (int i = 0; i < node.m_Successors.length; i++) {
					queueOfNodes.add(node.m_Successors[i]);
					queueOfWeights.add(a[index - 1] * weights[i]);
				}
			}
			return a;
		}
	}

	/**
	 * Returns the number of elements in the partition.
	 */
	@Override
	public int numElements() throws Exception {

		if (this.m_zeroR != null) {
			return 1;
		}
		return this.m_Tree.numNodes();
	}

	/**
	 * The inner class for dealing with the tree.
	 */
	public class Tree implements Serializable {

		/** For serialization */
		private static final long serialVersionUID = 3549573538656522569L;

		/** The subtrees appended to this tree. */
		protected Tree[] m_Successors;

		/** The attribute to split on. */
		protected int m_Attribute = -1;

		/** The split point. */
		protected double m_SplitPoint = Double.NaN;

		/** The proportions of training instances going down each branch. */
		protected double[] m_Prop = null;

		/**
		 * Class probabilities from the training data in the nominal case. Holds the mean in the numeric
		 * case.
		 */
		protected double[] m_ClassDistribution = null;

		/**
		 * Holds the sum of squared errors and the weight in the numeric case.
		 */
		protected double[] m_Distribution = null;

		/**
		 * Backfits the given data into the tree.
		 */
		public void backfitData(final Instances data) throws Exception {

			double totalWeight = 0;
			double totalSumSquared = 0;

			// Compute initial class counts
			double[] classProbs = new double[data.numClasses()];
			for (int i = 0; i < data.numInstances(); i++) {
				Instance inst = data.instance(i);
				if (data.classAttribute().isNominal()) {
					classProbs[(int) inst.classValue()] += inst.weight();
					totalWeight += inst.weight();
				} else {
					classProbs[0] += inst.classValue() * inst.weight();
					totalSumSquared += inst.classValue() * inst.classValue() * inst.weight();
					totalWeight += inst.weight();
				}
			}

			double trainVariance = 0;
			if (data.classAttribute().isNumeric()) {
				trainVariance = RandomTree.singleVariance(classProbs[0], totalSumSquared, totalWeight) / totalWeight;
				classProbs[0] /= totalWeight;
			}

			// Fit data into tree
			this.backfitData(data, classProbs, totalWeight);
		}

		/**
		 * Computes class distribution of an instance using the decision tree.
		 *
		 * @param instance
		 *          the instance to compute the distribution for
		 * @return the computed class distribution
		 * @throws Exception
		 *           if computation fails
		 */
		public double[] distributionForInstance(final Instance instance) throws Exception {

			double[] returnedDist = null;

			if (this.m_Attribute > -1) {

				// Node is not a leaf
				if (instance.isMissing(this.m_Attribute)) {

					// Value is missing
					returnedDist = new double[RandomTree.this.m_Info.numClasses()];

					// Split instance up
					for (int i = 0; i < this.m_Successors.length; i++) {
						double[] help = this.m_Successors[i].distributionForInstance(instance);
						if (help != null) {
							for (int j = 0; j < help.length; j++) {
								returnedDist[j] += this.m_Prop[i] * help[j];
							}
						}
					}
				} else if (RandomTree.this.m_Info.attribute(this.m_Attribute).isNominal()) {

					// For nominal attributes
					returnedDist = this.m_Successors[(int) instance.value(this.m_Attribute)].distributionForInstance(instance);
				} else {

					// For numeric attributes
					if (instance.value(this.m_Attribute) < this.m_SplitPoint) {
						returnedDist = this.m_Successors[0].distributionForInstance(instance);
					} else {
						returnedDist = this.m_Successors[1].distributionForInstance(instance);
					}
				}
			}

			// Node is a leaf or successor is empty?
			if ((this.m_Attribute == -1) || (returnedDist == null)) {

				// Is node empty?
				if (this.m_ClassDistribution == null) {
					if (RandomTree.this.getAllowUnclassifiedInstances()) {
						double[] result = new double[RandomTree.this.m_Info.numClasses()];
						if (RandomTree.this.m_Info.classAttribute().isNumeric()) {
							result[0] = Utils.missingValue();
						}
						return result;
					} else {
						return null;
					}
				}

				// Else return normalized distribution
				double[] normalizedDistribution = this.m_ClassDistribution.clone();
				if (RandomTree.this.m_Info.classAttribute().isNominal()) {
					Utils.normalize(normalizedDistribution);
				}
				return normalizedDistribution;
			} else {
				return returnedDist;
			}
		}

		/**
		 * Outputs one node for graph.
		 *
		 * @param text
		 *          the buffer to append the output to
		 * @param num
		 *          unique node id
		 * @return the next node id
		 * @throws Exception
		 *           if generation fails
		 */
		public int toGraph(final StringBuffer text, int num) throws Exception {

			int maxIndex = Utils.maxIndex(this.m_ClassDistribution);
			String classValue = RandomTree.this.m_Info.classAttribute().isNominal() ? RandomTree.this.m_Info.classAttribute().value(maxIndex) : Utils.doubleToString(this.m_ClassDistribution[0], RandomTree.this.getNumDecimalPlaces());

			num++;
			if (this.m_Attribute == -1) {
				text.append("N" + Integer.toHexString(this.hashCode()) + " [label=\"" + num + ": " + classValue + "\"" + "shape=box]\n");
			} else {
				text.append("N" + Integer.toHexString(this.hashCode()) + " [label=\"" + num + ": " + classValue + "\"]\n");
				for (int i = 0; i < this.m_Successors.length; i++) {
					text.append("N" + Integer.toHexString(this.hashCode()) + "->" + "N" + Integer.toHexString(this.m_Successors[i].hashCode()) + " [label=\"" + RandomTree.this.m_Info.attribute(this.m_Attribute).name());
					if (RandomTree.this.m_Info.attribute(this.m_Attribute).isNumeric()) {
						if (i == 0) {
							text.append(" < " + Utils.doubleToString(this.m_SplitPoint, RandomTree.this.getNumDecimalPlaces()));
						} else {
							text.append(" >= " + Utils.doubleToString(this.m_SplitPoint, RandomTree.this.getNumDecimalPlaces()));
						}
					} else {
						text.append(" = " + RandomTree.this.m_Info.attribute(this.m_Attribute).value(i));
					}
					text.append("\"]\n");
					num = this.m_Successors[i].toGraph(text, num);
				}
			}

			return num;
		}

		/**
		 * Outputs a leaf.
		 *
		 * @return the leaf as string
		 * @throws Exception
		 *           if generation fails
		 */
		protected String leafString() throws Exception {

			double sum = 0, maxCount = 0;
			int maxIndex = 0;
			double classMean = 0;
			double avgError = 0;
			if (this.m_ClassDistribution != null) {
				if (RandomTree.this.m_Info.classAttribute().isNominal()) {
					sum = Utils.sum(this.m_ClassDistribution);
					maxIndex = Utils.maxIndex(this.m_ClassDistribution);
					maxCount = this.m_ClassDistribution[maxIndex];
				} else {
					classMean = this.m_ClassDistribution[0];
					if (this.m_Distribution[1] > 0) {
						avgError = this.m_Distribution[0] / this.m_Distribution[1];
					}
				}
			}

			if (RandomTree.this.m_Info.classAttribute().isNumeric()) {
				return " : " + Utils.doubleToString(classMean, RandomTree.this.getNumDecimalPlaces()) + " (" + Utils.doubleToString(this.m_Distribution[1], RandomTree.this.getNumDecimalPlaces()) + "/"
						+ Utils.doubleToString(avgError, RandomTree.this.getNumDecimalPlaces()) + ")";
			}

			return " : " + RandomTree.this.m_Info.classAttribute().value(maxIndex) + " (" + Utils.doubleToString(sum, RandomTree.this.getNumDecimalPlaces()) + "/" + Utils.doubleToString(sum - maxCount, RandomTree.this.getNumDecimalPlaces())
					+ ")";
		}

		/**
		 * Recursively outputs the tree.
		 *
		 * @param level
		 *          the current level of the tree
		 * @return the generated subtree
		 */
		protected String toString(final int level) {

			try {
				StringBuffer text = new StringBuffer();

				if (this.m_Attribute == -1) {

					// Output leaf info
					return this.leafString();
				} else if (RandomTree.this.m_Info.attribute(this.m_Attribute).isNominal()) {

					// For nominal attributes
					for (int i = 0; i < this.m_Successors.length; i++) {
						text.append("\n");
						for (int j = 0; j < level; j++) {
							text.append("|   ");
						}
						text.append(RandomTree.this.m_Info.attribute(this.m_Attribute).name() + " = " + RandomTree.this.m_Info.attribute(this.m_Attribute).value(i));
						text.append(this.m_Successors[i].toString(level + 1));
					}
				} else {

					// For numeric attributes
					text.append("\n");
					for (int j = 0; j < level; j++) {
						text.append("|   ");
					}
					text.append(RandomTree.this.m_Info.attribute(this.m_Attribute).name() + " < " + Utils.doubleToString(this.m_SplitPoint, RandomTree.this.getNumDecimalPlaces()));
					text.append(this.m_Successors[0].toString(level + 1));
					text.append("\n");
					for (int j = 0; j < level; j++) {
						text.append("|   ");
					}
					text.append(RandomTree.this.m_Info.attribute(this.m_Attribute).name() + " >= " + Utils.doubleToString(this.m_SplitPoint, RandomTree.this.getNumDecimalPlaces()));
					text.append(this.m_Successors[1].toString(level + 1));
				}

				return text.toString();
			} catch (Exception e) {
				e.printStackTrace();
				return "RandomTree: tree can't be printed";
			}
		}

		/**
		 * Recursively backfits data into the tree.
		 *
		 * @param data
		 *          the data to work with
		 * @param classProbs
		 *          the class distribution
		 * @throws Exception
		 *           if generation fails
		 */
		protected void backfitData(final Instances data, final double[] classProbs, final double totalWeight) throws Exception {

			// Make leaf if there are no training instances
			if (data.numInstances() == 0) {
				this.m_Attribute = -1;
				this.m_ClassDistribution = null;
				if (data.classAttribute().isNumeric()) {
					this.m_Distribution = new double[2];
				}
				this.m_Prop = null;
				return;
			}

			double priorVar = 0;
			if (data.classAttribute().isNumeric()) {

				// Compute prior variance
				double totalSum = 0, totalSumSquared = 0, totalSumOfWeights = 0;
				for (int i = 0; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					totalSum += inst.classValue() * inst.weight();
					totalSumSquared += inst.classValue() * inst.classValue() * inst.weight();
					totalSumOfWeights += inst.weight();
				}
				priorVar = RandomTree.singleVariance(totalSum, totalSumSquared, totalSumOfWeights);
			}

			// Check if node doesn't contain enough instances or is pure
			// or maximum depth reached
			this.m_ClassDistribution = classProbs.clone();

			/*
			 * if (Utils.sum(m_ClassDistribution) < 2 * m_MinNum ||
			 * Utils.eq(m_ClassDistribution[Utils.maxIndex(m_ClassDistribution)], Utils
			 * .sum(m_ClassDistribution))) {
			 *
			 * // Make leaf m_Attribute = -1; m_Prop = null; return; }
			 */

			// Are we at an inner node
			if (this.m_Attribute > -1) {

				// Compute new weights for subsets based on backfit data
				this.m_Prop = new double[this.m_Successors.length];
				for (int i = 0; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					if (!inst.isMissing(this.m_Attribute)) {
						if (data.attribute(this.m_Attribute).isNominal()) {
							this.m_Prop[(int) inst.value(this.m_Attribute)] += inst.weight();
						} else {
							this.m_Prop[(inst.value(this.m_Attribute) < this.m_SplitPoint) ? 0 : 1] += inst.weight();
						}
					}
				}

				// If we only have missing values we can make this node into a leaf
				if (Utils.sum(this.m_Prop) <= 0) {
					this.m_Attribute = -1;
					this.m_Prop = null;

					if (data.classAttribute().isNumeric()) {
						this.m_Distribution = new double[2];
						this.m_Distribution[0] = priorVar;
						this.m_Distribution[1] = totalWeight;
					}

					return;
				}

				// Otherwise normalize the proportions
				Utils.normalize(this.m_Prop);

				// Split data
				Instances[] subsets = this.splitData(data);

				// Go through subsets
				for (int i = 0; i < subsets.length; i++) {

					// Compute distribution for current subset
					double[] dist = new double[data.numClasses()];
					double sumOfWeights = 0;
					for (int j = 0; j < subsets[i].numInstances(); j++) {
						if (data.classAttribute().isNominal()) {
							dist[(int) subsets[i].instance(j).classValue()] += subsets[i].instance(j).weight();
						} else {
							dist[0] += subsets[i].instance(j).classValue() * subsets[i].instance(j).weight();
							sumOfWeights += subsets[i].instance(j).weight();
						}
					}

					if (sumOfWeights > 0) {
						dist[0] /= sumOfWeights;
					}

					// Backfit subset
					this.m_Successors[i].backfitData(subsets[i], dist, totalWeight);
				}

				// If unclassified instances are allowed, we don't need to store the
				// class distribution
				if (RandomTree.this.getAllowUnclassifiedInstances()) {
					this.m_ClassDistribution = null;
					return;
				}

				for (int i = 0; i < subsets.length; i++) {
					if (this.m_Successors[i].m_ClassDistribution == null) {
						return;
					}
				}
				this.m_ClassDistribution = null;

				// If we have a least two non-empty successors, we should keep this tree
				/*
				 * int nonEmptySuccessors = 0; for (int i = 0; i < subsets.length; i++) { if
				 * (m_Successors[i].m_ClassDistribution != null) { nonEmptySuccessors++; if (nonEmptySuccessors > 1)
				 * { return; } } }
				 *
				 * // Otherwise, this node is a leaf or should become a leaf m_Successors = null; m_Attribute = -1;
				 * m_Prop = null; return;
				 */
			}
		}

		/**
		 * Recursively generates a tree.
		 *
		 * @param data
		 *          the data to work with
		 * @param classProbs
		 *          the class distribution
		 * @param attIndicesWindow
		 *          the attribute window to choose attributes from
		 * @param random
		 *          random number generator for choosing random attributes
		 * @param depth
		 *          the current depth
		 * @throws Exception
		 *           if generation fails
		 */
		protected void buildTree(final Instances data, final double[] classProbs, final int[] attIndicesWindow, double totalWeight, final Random random, final int depth, final double minVariance) throws Exception {

			// Make leaf if there are no training instances
			if (data.numInstances() == 0) {
				this.m_Attribute = -1;
				this.m_ClassDistribution = null;
				this.m_Prop = null;

				if (data.classAttribute().isNumeric()) {
					this.m_Distribution = new double[2];
				}
				return;
			}

			double priorVar = 0;
			if (data.classAttribute().isNumeric()) {

				// Compute prior variance
				double totalSum = 0, totalSumSquared = 0, totalSumOfWeights = 0;
				for (int i = 0; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					totalSum += inst.classValue() * inst.weight();
					totalSumSquared += inst.classValue() * inst.classValue() * inst.weight();
					totalSumOfWeights += inst.weight();
				}
				priorVar = RandomTree.singleVariance(totalSum, totalSumSquared, totalSumOfWeights);
			}

			// Check if node doesn't contain enough instances or is pure
			// or maximum depth reached
			if (data.classAttribute().isNominal()) {
				totalWeight = Utils.sum(classProbs);
			}
			// System.err.println("Total weight " + totalWeight);
			// double sum = Utils.sum(classProbs);
			if (totalWeight < 2 * RandomTree.this.m_MinNum ||

			// Nominal case
					(data.classAttribute().isNominal() && Utils.eq(classProbs[Utils.maxIndex(classProbs)], Utils.sum(classProbs)))

					||

					// Numeric case
					(data.classAttribute().isNumeric() && priorVar / totalWeight < minVariance)

					||

					// check tree depth
					((RandomTree.this.getMaxDepth() > 0) && (depth >= RandomTree.this.getMaxDepth()))) {

				// Make leaf
				this.m_Attribute = -1;
				this.m_ClassDistribution = classProbs.clone();
				if (data.classAttribute().isNumeric()) {
					this.m_Distribution = new double[2];
					this.m_Distribution[0] = priorVar;
					this.m_Distribution[1] = totalWeight;
				}

				this.m_Prop = null;
				return;
			}

			// Compute class distributions and value of splitting
			// criterion for each attribute
			double val = -Double.MAX_VALUE;
			double split = -Double.MAX_VALUE;
			double[][] bestDists = null;
			double[] bestProps = null;
			int bestIndex = 0;

			// Handles to get arrays out of distribution method
			double[][] props = new double[1][0];
			double[][][] dists = new double[1][0][0];
			double[][] totalSubsetWeights = new double[data.numAttributes()][0];

			// Investigate K random attributes
			int attIndex = 0;
			int windowSize = attIndicesWindow.length;
			int k = RandomTree.this.m_KValue;
			boolean gainFound = false;
			double[] tempNumericVals = new double[data.numAttributes()];
			while ((windowSize > 0) && (k-- > 0 || !gainFound)) {

				int chosenIndex = random.nextInt(windowSize);
				attIndex = attIndicesWindow[chosenIndex];

				// shift chosen attIndex out of window
				attIndicesWindow[chosenIndex] = attIndicesWindow[windowSize - 1];
				attIndicesWindow[windowSize - 1] = attIndex;
				windowSize--;

				double currSplit = data.classAttribute().isNominal() ? this.distribution(props, dists, attIndex, data) : this.numericDistribution(props, dists, attIndex, totalSubsetWeights, data, tempNumericVals);

				double currVal = data.classAttribute().isNominal() ? this.gain(dists[0], this.priorVal(dists[0])) : tempNumericVals[attIndex];

				if (Utils.gr(currVal, 0)) {
					gainFound = true;
				}

				if ((currVal > val) || ((!RandomTree.this.getBreakTiesRandomly()) && (currVal == val) && (attIndex < bestIndex))) {
					val = currVal;
					bestIndex = attIndex;
					split = currSplit;
					bestProps = props[0];
					bestDists = dists[0];
				}
			}

			// Find best attribute
			this.m_Attribute = bestIndex;

			// Any useful split found?
			if (Utils.gr(val, 0)) {
				if (RandomTree.this.m_computeImpurityDecreases) {
					RandomTree.this.m_impurityDecreasees[this.m_Attribute][0] += val;
					RandomTree.this.m_impurityDecreasees[this.m_Attribute][1]++;
				}

				// Build subtrees
				this.m_SplitPoint = split;
				this.m_Prop = bestProps;
				Instances[] subsets = this.splitData(data);
				this.m_Successors = new Tree[bestDists.length];
				double[] attTotalSubsetWeights = totalSubsetWeights[bestIndex];

				for (int i = 0; i < bestDists.length; i++) {
					this.m_Successors[i] = new Tree();
					this.m_Successors[i].buildTree(subsets[i], bestDists[i], attIndicesWindow, data.classAttribute().isNominal() ? 0 : attTotalSubsetWeights[i], random, depth + 1, minVariance);
				}

				// If all successors are non-empty, we don't need to store the class
				// distribution
				boolean emptySuccessor = false;
				for (int i = 0; i < subsets.length; i++) {
					if (this.m_Successors[i].m_ClassDistribution == null) {
						emptySuccessor = true;
						break;
					}
				}
				if (emptySuccessor) {
					this.m_ClassDistribution = classProbs.clone();
				}
			} else {

				// Make leaf
				this.m_Attribute = -1;
				this.m_ClassDistribution = classProbs.clone();
				if (data.classAttribute().isNumeric()) {
					this.m_Distribution = new double[2];
					this.m_Distribution[0] = priorVar;
					this.m_Distribution[1] = totalWeight;
				}
			}
		}

		/**
		 * Computes size of the tree.
		 *
		 * @return the number of nodes
		 */
		public int numNodes() {

			if (this.m_Attribute == -1) {
				return 1;
			} else {
				int size = 1;
				for (Tree m_Successor : this.m_Successors) {
					size += m_Successor.numNodes();
				}
				return size;
			}
		}

		/**
		 * Splits instances into subsets based on the given split.
		 *
		 * @param data
		 *          the data to work with
		 * @return the subsets of instances
		 * @throws Exception
		 *           if something goes wrong
		 */
		protected Instances[] splitData(final Instances data) throws Exception {

			// Allocate array of Instances objects
			Instances[] subsets = new Instances[this.m_Prop.length];
			for (int i = 0; i < this.m_Prop.length; i++) {
				subsets[i] = new Instances(data, data.numInstances());
			}

			// Go through the data
			for (int i = 0; i < data.numInstances(); i++) {

				// Get instance
				Instance inst = data.instance(i);

				// Does the instance have a missing value?
				if (inst.isMissing(this.m_Attribute)) {

					// Split instance up
					for (int k = 0; k < this.m_Prop.length; k++) {
						if (this.m_Prop[k] > 0) {
							Instance copy = (Instance) inst.copy();
							copy.setWeight(this.m_Prop[k] * inst.weight());
							subsets[k].add(copy);
						}
					}

					// Proceed to next instance
					continue;
				}

				// Do we have a nominal attribute?
				if (data.attribute(this.m_Attribute).isNominal()) {
					subsets[(int) inst.value(this.m_Attribute)].add(inst);

					// Proceed to next instance
					continue;
				}

				// Do we have a numeric attribute?
				if (data.attribute(this.m_Attribute).isNumeric()) {
					subsets[(inst.value(this.m_Attribute) < this.m_SplitPoint) ? 0 : 1].add(inst);

					// Proceed to next instance
					continue;
				}

				// Else throw an exception
				throw new IllegalArgumentException("Unknown attribute type");
			}

			// Save memory
			for (int i = 0; i < this.m_Prop.length; i++) {
				subsets[i].compactify();
			}

			// Return the subsets
			return subsets;
		}

		/**
		 * Computes numeric class distribution for an attribute
		 *
		 * @param props
		 * @param dists
		 * @param att
		 * @param subsetWeights
		 * @param data
		 * @param vals
		 * @return
		 * @throws Exception
		 *           if a problem occurs
		 */
		protected double numericDistribution(final double[][] props, final double[][][] dists, final int att, final double[][] subsetWeights, final Instances data, final double[] vals) throws Exception {

			double splitPoint = Double.NaN;
			Attribute attribute = data.attribute(att);
			double[][] dist = null;
			double[] sums = null;
			double[] sumSquared = null;
			double[] sumOfWeights = null;
			double totalSum = 0, totalSumSquared = 0, totalSumOfWeights = 0;
			int indexOfFirstMissingValue = data.numInstances();

			if (attribute.isNominal()) {
				sums = new double[attribute.numValues()];
				sumSquared = new double[attribute.numValues()];
				sumOfWeights = new double[attribute.numValues()];
				int attVal;

				for (int i = 0; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					if (inst.isMissing(att)) {

						// Skip missing values at this stage
						if (indexOfFirstMissingValue == data.numInstances()) {
							indexOfFirstMissingValue = i;
						}
						continue;
					}

					attVal = (int) inst.value(att);
					sums[attVal] += inst.classValue() * inst.weight();
					sumSquared[attVal] += inst.classValue() * inst.classValue() * inst.weight();
					sumOfWeights[attVal] += inst.weight();
				}

				totalSum = Utils.sum(sums);
				totalSumSquared = Utils.sum(sumSquared);
				totalSumOfWeights = Utils.sum(sumOfWeights);
			} else {
				// For numeric attributes
				sums = new double[2];
				sumSquared = new double[2];
				sumOfWeights = new double[2];
				double[] currSums = new double[2];
				double[] currSumSquared = new double[2];
				double[] currSumOfWeights = new double[2];

				// Sort data
				data.sort(att);

				// Move all instances into second subset
				for (int j = 0; j < data.numInstances(); j++) {
					Instance inst = data.instance(j);
					if (inst.isMissing(att)) {

						// Can stop as soon as we hit a missing value
						indexOfFirstMissingValue = j;
						break;
					}

					currSums[1] += inst.classValue() * inst.weight();
					currSumSquared[1] += inst.classValue() * inst.classValue() * inst.weight();
					currSumOfWeights[1] += inst.weight();
				}

				totalSum = currSums[1];
				totalSumSquared = currSumSquared[1];
				totalSumOfWeights = currSumOfWeights[1];

				sums[1] = currSums[1];
				sumSquared[1] = currSumSquared[1];
				sumOfWeights[1] = currSumOfWeights[1];

				// Try all possible split points
				double currSplit = data.instance(0).value(att);
				double currVal, bestVal = Double.MAX_VALUE;

				for (int i = 0; i < indexOfFirstMissingValue; i++) {
					Instance inst = data.instance(i);

					if (inst.value(att) > currSplit) {
						currVal = RandomTree.variance(currSums, currSumSquared, currSumOfWeights);
						if (currVal < bestVal) {
							bestVal = currVal;
							splitPoint = (inst.value(att) + currSplit) / 2.0;

							// Check for numeric precision problems
							if (splitPoint <= currSplit) {
								splitPoint = inst.value(att);
							}

							for (int j = 0; j < 2; j++) {
								sums[j] = currSums[j];
								sumSquared[j] = currSumSquared[j];
								sumOfWeights[j] = currSumOfWeights[j];
							}
						}
					}

					currSplit = inst.value(att);

					double classVal = inst.classValue() * inst.weight();
					double classValSquared = inst.classValue() * classVal;

					currSums[0] += classVal;
					currSumSquared[0] += classValSquared;
					currSumOfWeights[0] += inst.weight();

					currSums[1] -= classVal;
					currSumSquared[1] -= classValSquared;
					currSumOfWeights[1] -= inst.weight();
				}
			}

			// Compute weights
			props[0] = new double[sums.length];
			for (int k = 0; k < props[0].length; k++) {
				props[0][k] = sumOfWeights[k];
			}
			if (!(Utils.sum(props[0]) > 0)) {
				for (int k = 0; k < props[0].length; k++) {
					props[0][k] = 1.0 / props[0].length;
				}
			} else {
				Utils.normalize(props[0]);
			}

			// Distribute weights for instances with missing values
			for (int i = indexOfFirstMissingValue; i < data.numInstances(); i++) {
				Instance inst = data.instance(i);

				for (int j = 0; j < sums.length; j++) {
					sums[j] += props[0][j] * inst.classValue() * inst.weight();
					sumSquared[j] += props[0][j] * inst.classValue() * inst.classValue() * inst.weight();
					sumOfWeights[j] += props[0][j] * inst.weight();
				}
				totalSum += inst.classValue() * inst.weight();
				totalSumSquared += inst.classValue() * inst.classValue() * inst.weight();
				totalSumOfWeights += inst.weight();
			}

			// Compute final distribution
			dist = new double[sums.length][data.numClasses()];
			for (int j = 0; j < sums.length; j++) {
				if (sumOfWeights[j] > 0) {
					dist[j][0] = sums[j] / sumOfWeights[j];
				} else {
					dist[j][0] = totalSum / totalSumOfWeights;
				}
			}

			// Compute variance gain
			double priorVar = singleVariance(totalSum, totalSumSquared, totalSumOfWeights);
			double var = variance(sums, sumSquared, sumOfWeights);
			double gain = priorVar - var;

			// Return distribution and split point
			subsetWeights[att] = sumOfWeights;
			dists[0] = dist;
			vals[att] = gain;

			return splitPoint;
		}

		/**
		 * Computes class distribution for an attribute.
		 *
		 * @param props
		 * @param dists
		 * @param att
		 *          the attribute index
		 * @param data
		 *          the data to work with
		 * @throws Exception
		 *           if something goes wrong
		 */
		protected double distribution(final double[][] props, final double[][][] dists, final int att, final Instances data) throws Exception {

			double splitPoint = Double.NaN;
			Attribute attribute = data.attribute(att);
			double[][] dist = null;
			int indexOfFirstMissingValue = data.numInstances();

			if (attribute.isNominal()) {

				// For nominal attributes
				dist = new double[attribute.numValues()][data.numClasses()];
				for (int i = 0; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					if (inst.isMissing(att)) {

						// Skip missing values at this stage
						if (indexOfFirstMissingValue == data.numInstances()) {
							indexOfFirstMissingValue = i;
						}
						continue;
					}
					dist[(int) inst.value(att)][(int) inst.classValue()] += inst.weight();
				}
			} else {

				// For numeric attributes
				double[][] currDist = new double[2][data.numClasses()];
				dist = new double[2][data.numClasses()];

				// Sort data
				data.sort(att);

				// Move all instances into second subset
				for (int j = 0; j < data.numInstances(); j++) {
					Instance inst = data.instance(j);
					if (inst.isMissing(att)) {

						// Can stop as soon as we hit a missing value
						indexOfFirstMissingValue = j;
						break;
					}
					currDist[1][(int) inst.classValue()] += inst.weight();
				}

				// Value before splitting
				double priorVal = this.priorVal(currDist);

				// Save initial distribution
				for (int j = 0; j < currDist.length; j++) {
					System.arraycopy(currDist[j], 0, dist[j], 0, dist[j].length);
				}

				// Try all possible split points
				double currSplit = data.instance(0).value(att);
				double currVal, bestVal = -Double.MAX_VALUE;
				for (int i = 0; i < indexOfFirstMissingValue; i++) {
					Instance inst = data.instance(i);
					double attVal = inst.value(att);

					// Can we place a sensible split point here?
					if (attVal > currSplit) {

						// Compute gain for split point
						currVal = this.gain(currDist, priorVal);

						// Is the current split point the best point so far?
						if (currVal > bestVal) {

							// Store value of current point
							bestVal = currVal;

							// Save split point
							splitPoint = (attVal + currSplit) / 2.0;

							// Check for numeric precision problems
							if (splitPoint <= currSplit) {
								splitPoint = attVal;
							}

							// Save distribution
							for (int j = 0; j < currDist.length; j++) {
								System.arraycopy(currDist[j], 0, dist[j], 0, dist[j].length);
							}
						}

						// Update value
						currSplit = attVal;
					}

					// Shift over the weight
					int classVal = (int) inst.classValue();
					currDist[0][classVal] += inst.weight();
					currDist[1][classVal] -= inst.weight();
				}
			}

			// Compute weights for subsets
			props[0] = new double[dist.length];
			for (int k = 0; k < props[0].length; k++) {
				props[0][k] = Utils.sum(dist[k]);
			}
			if (Utils.eq(Utils.sum(props[0]), 0)) {
				for (int k = 0; k < props[0].length; k++) {
					props[0][k] = 1.0 / props[0].length;
				}
			} else {
				Utils.normalize(props[0]);
			}

			// Distribute weights for instances with missing values
			for (int i = indexOfFirstMissingValue; i < data.numInstances(); i++) {
				Instance inst = data.instance(i);
				if (attribute.isNominal()) {

					// Need to check if attribute value is missing
					if (inst.isMissing(att)) {
						for (int j = 0; j < dist.length; j++) {
							dist[j][(int) inst.classValue()] += props[0][j] * inst.weight();
						}
					}
				} else {

					// Can be sure that value is missing, so no test required
					for (int j = 0; j < dist.length; j++) {
						dist[j][(int) inst.classValue()] += props[0][j] * inst.weight();
					}
				}
			}

			// Return distribution and split point
			dists[0] = dist;
			return splitPoint;
		}

		/**
		 * Computes value of splitting criterion before split.
		 *
		 * @param dist
		 *          the distributions
		 * @return the splitting criterion
		 * @throws InterruptedException
		 */
		protected double priorVal(final double[][] dist) throws InterruptedException {

			return ContingencyTables.entropyOverColumns(dist);
		}

		/**
		 * Computes value of splitting criterion after split.
		 *
		 * @param dist
		 *          the distributions
		 * @param priorVal
		 *          the splitting criterion
		 * @return the gain after the split
		 */
		protected double gain(final double[][] dist, final double priorVal) {

			return priorVal - ContingencyTables.entropyConditionedOnRows(dist);
		}

		/**
		 * Returns the revision string.
		 *
		 * @return the revision
		 */
		public String getRevision() {
			return RevisionUtils.extract("$Revision$");
		}

		/**
		 * Outputs one node for graph.
		 *
		 * @param text
		 *          the buffer to append the output to
		 * @param num
		 *          the current node id
		 * @param parent
		 *          the parent of the nodes
		 * @return the next node id
		 * @throws Exception
		 *           if something goes wrong
		 */
		protected int toGraph(final StringBuffer text, int num, final Tree parent) throws Exception {

			num++;
			if (this.m_Attribute == -1) {
				text.append("N" + Integer.toHexString(Tree.this.hashCode()) + " [label=\"" + num + Utils.backQuoteChars(this.leafString()) + "\"" + " shape=box]\n");

			} else {
				text.append("N" + Integer.toHexString(Tree.this.hashCode()) + " [label=\"" + num + ": " + Utils.backQuoteChars(RandomTree.this.m_Info.attribute(this.m_Attribute).name()) + "\"]\n");
				for (int i = 0; i < this.m_Successors.length; i++) {
					text.append("N" + Integer.toHexString(Tree.this.hashCode()) + "->" + "N" + Integer.toHexString(this.m_Successors[i].hashCode()) + " [label=\"");
					if (RandomTree.this.m_Info.attribute(this.m_Attribute).isNumeric()) {
						if (i == 0) {
							text.append(" < " + Utils.doubleToString(this.m_SplitPoint, RandomTree.this.getNumDecimalPlaces()));
						} else {
							text.append(" >= " + Utils.doubleToString(this.m_SplitPoint, RandomTree.this.getNumDecimalPlaces()));
						}
					} else {
						text.append(" = " + Utils.backQuoteChars(RandomTree.this.m_Info.attribute(this.m_Attribute).value(i)));
					}
					text.append("\"]\n");
					num = this.m_Successors[i].toGraph(text, num, this);
				}
			}

			return num;
		}

		/**
		 * Get the successor subtrees of this tree.
		 *
		 * @return the subtrees
		 */
		public Tree[] getM_Successors() {
			return this.m_Successors;
		}

		/**
		 * Get the attribute this tree splits by.
		 *
		 * @return the attribute index
		 */
		public int getM_Attribute() {
			return this.m_Attribute;
		}

		/**
		 * Get the split point for the attribute (relevant only if it is numeric). If the attribute value is strictly smaller
		 * than the split point, the relevant successor is the first successor, otherwise it is the second successor.
		 *
		 * @return the split point
		 */
		public double getM_SplitPoint() {
			return this.m_SplitPoint;
		}
		
		/**
		 * Gets the class distribution for the last Instances object
		 * 
		 * @return the array that captures the likelihood of each class
		 */
		public double[] getM_Classdistribution() {
			return this.m_ClassDistribution;
		}
	}

	/**
	 * Computes variance for subsets.
	 *
	 * @param s
	 * @param sS
	 * @param sumOfWeights
	 * @return the variance
	 */
	protected static double variance(final double[] s, final double[] sS, final double[] sumOfWeights) {

		double var = 0;

		for (int i = 0; i < s.length; i++) {
			if (sumOfWeights[i] > 0) {
				var += singleVariance(s[i], sS[i], sumOfWeights[i]);
			}
		}

		return var;
	}

	/**
	 * Computes the variance for a single set
	 *
	 * @param s
	 * @param sS
	 * @param weight
	 *          the weight
	 * @return the variance
	 */
	protected static double singleVariance(final double s, final double sS, final double weight) {

		return sS - ((s * s) / weight);
	}

	/**
	 * Main method for this class.
	 *
	 * @param argv
	 *          the commandline parameters
	 */
	public static void main(final String[] argv) {
		runClassifier(new RandomTree(), argv);
	}

	public Tree getM_Tree() {
		return this.m_Tree;
	}
}