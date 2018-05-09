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
 *    LMTNode.java
 *    Copyright (C) 2003-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.lmt;

import java.util.Collections;
import java.util.Comparator;
import java.util.Vector;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

/**
 * Auxiliary class for list of LMTNodes
 */
class CompareNode implements Comparator<LMTNode>, RevisionHandler {

  /**
   * Compares its two arguments for order.
   *
   * @param o1
   *          first object
   * @param o2
   *          second object
   * @return a negative integer, zero, or a positive integer as the first argument is less than, equal
   *         to, or greater than the second.
   */
  @Override
  public int compare(final LMTNode o1, final LMTNode o2) {
    if (o1.m_alpha < o2.m_alpha) {
      return -1;
    }
    if (o1.m_alpha > o2.m_alpha) {
      return 1;
    }
    return 0;
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

/**
 * Class for logistic model tree structure.
 *
 *
 * @author Niels Landwehr
 * @author Marc Sumner
 * @version $Revision$
 */
public class LMTNode extends LogisticBase {

  /** for serialization */
  static final long serialVersionUID = 1862737145870398755L;

  /** Total number of training instances. */
  protected double m_totalInstanceWeight;

  /** Node id */
  protected int m_id;

  /** ID of logistic model at leaf */
  protected int m_leafModelNum;

  /** Alpha-value (for pruning) at the node */
  public double m_alpha;

  /**
   * Weighted number of training examples currently misclassified by the logistic model at the node
   */
  public double m_numIncorrectModel;

  /**
   * Weighted number of training examples currently misclassified by the subtree rooted at the node
   */
  public double m_numIncorrectTree;

  /** minimum number of instances at which a node is considered for splitting */
  protected int m_minNumInstances;

  /** ModelSelection object (for splitting) */
  protected ModelSelection m_modelSelection;

  /** Filter to convert nominal attributes to binary */
  protected NominalToBinary m_nominalToBinary;

  /** Number of folds for CART pruning */
  protected static int m_numFoldsPruning = 5;

  /**
   * Use heuristic that determines the number of LogitBoost iterations only once in the beginning?
   */
  protected boolean m_fastRegression;

  /** Number of instances at the node */
  protected int m_numInstances;

  /** The ClassifierSplitModel (for splitting) */
  protected ClassifierSplitModel m_localModel;

  /** Array of children of the node */
  protected LMTNode[] m_sons;

  /** True if node is leaf */
  protected boolean m_isLeaf;

  /**
   * Constructor for logistic model tree node.
   *
   * @param modelSelection
   *          selection method for local splitting model
   * @param numBoostingIterations
   *          sets the numBoostingIterations parameter
   * @param fastRegression
   *          sets the fastRegression parameter
   * @param errorOnProbabilities
   *          Use error on probabilities for stopping criterion of LogitBoost?
   * @param minNumInstances
   *          minimum number of instances at which a node is considered for splitting
   */
  public LMTNode(final ModelSelection modelSelection, final int numBoostingIterations, final boolean fastRegression, final boolean errorOnProbabilities, final int minNumInstances,
      final double weightTrimBeta, final boolean useAIC, final NominalToBinary ntb, final int numDecimalPlaces) {
    this.m_modelSelection = modelSelection;
    this.m_fixedNumIterations = numBoostingIterations;
    this.m_fastRegression = fastRegression;
    this.m_errorOnProbabilities = errorOnProbabilities;
    this.m_minNumInstances = minNumInstances;
    this.m_maxIterations = 200;
    this.setWeightTrimBeta(weightTrimBeta);
    this.setUseAIC(useAIC);
    this.m_nominalToBinary = ntb;
    this.m_numDecimalPlaces = numDecimalPlaces;
  }

  /**
   * Method for building a logistic model tree (only called for the root node). Grows an initial
   * logistic model tree and prunes it back using the CART pruning scheme.
   *
   * @param data
   *          the data to train with
   * @throws Exception
   *           if something goes wrong
   */
  @Override
  public void buildClassifier(final Instances data) throws Exception {

    // heuristic to avoid cross-validating the number of LogitBoost iterations
    // at every node: build standalone logistic model and take its optimum
    // number
    // of iteration everywhere in the tree.
    if (this.m_fastRegression && (this.m_fixedNumIterations < 0)) {
      this.m_fixedNumIterations = this.tryLogistic(data);
    }

    // Need to cross-validate alpha-parameter for CART-pruning
    Instances cvData = new Instances(data);
    cvData.stratify(m_numFoldsPruning);

    double[][] alphas = new double[m_numFoldsPruning][];
    double[][] errors = new double[m_numFoldsPruning][];

    for (int i = 0; i < m_numFoldsPruning; i++) {
      // XXX kill weka execution
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
      }
      // for every fold, grow tree on training set...
      Instances train = cvData.trainCV(m_numFoldsPruning, i);
      Instances test = cvData.testCV(m_numFoldsPruning, i);

      this.buildTree(train, null, train.numInstances(), 0, null);

      int numNodes = this.getNumInnerNodes();
      alphas[i] = new double[numNodes + 2];
      errors[i] = new double[numNodes + 2];

      // ... then prune back and log alpha-values and errors on test set
      this.prune(alphas[i], errors[i], test);
    }

    // don't need CV data anymore
    cvData = null;

    // build tree using all the data
    this.buildTree(data, null, data.numInstances(), 0, null);
    int numNodes = this.getNumInnerNodes();

    double[] treeAlphas = new double[numNodes + 2];

    // prune back and log alpha-values
    int iterations = this.prune(treeAlphas, null, null);

    double[] treeErrors = new double[numNodes + 2];

    for (int i = 0; i <= iterations; i++) {
      // XXX kill weka execution
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
      }
      // compute midpoint alphas
      double alpha = Math.sqrt(treeAlphas[i] * treeAlphas[i + 1]);
      double error = 0;

      // compute error estimate for final trees from the midpoint-alphas and the
      // error estimates gotten in
      // the cross-validation
      for (int k = 0; k < m_numFoldsPruning; k++) {
        int l = 0;
        while (alphas[k][l] <= alpha) {
          l++;
        }
        error += errors[k][l - 1];
      }

      treeErrors[i] = error;
    }

    // find best alpha
    int best = -1;
    double bestError = Double.MAX_VALUE;
    for (int i = iterations; i >= 0; i--) {
      // XXX kill weka execution
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
      }
      if (treeErrors[i] < bestError) {
        bestError = treeErrors[i];
        best = i;
      }
    }

    double bestAlpha = Math.sqrt(treeAlphas[best] * treeAlphas[best + 1]);

    // "unprune" final tree (faster than regrowing it)
    this.unprune();

    // CART-prune it with best alpha
    this.prune(bestAlpha);
  }

  /**
   * Method for building the tree structure. Builds a logistic model, splits the node and recursively
   * builds tree for child nodes.
   *
   * @param data
   *          the training data passed on to this node
   * @param higherRegressions
   *          An array of regression functions produced by LogitBoost at higher levels in the tree.
   *          They represent a logistic regression model that is refined locally at this node.
   * @param totalInstanceWeight
   *          the total number of training examples
   * @param higherNumParameters
   *          effective number of parameters in the logistic regression model built in parent nodes
   * @throws Exception
   *           if something goes wrong
   */
  public void buildTree(final Instances data, final SimpleLinearRegression[][] higherRegressions, final double totalInstanceWeight, final double higherNumParameters,
      final Instances numericDataHeader) throws Exception {

    // save some stuff
    this.m_totalInstanceWeight = totalInstanceWeight;
    this.m_train = data; // no need to copy the data here

    this.m_isLeaf = true;
    this.m_sons = null;

    this.m_numInstances = this.m_train.numInstances();
    this.m_numClasses = this.m_train.numClasses();

    // init
    this.m_numericDataHeader = numericDataHeader;
    this.m_numericData = this.getNumericData(this.m_train);

    if (higherRegressions == null) {
      this.m_regressions = this.initRegressions();
    } else {
      this.m_regressions = higherRegressions;
    }

    this.m_numParameters = higherNumParameters;
    this.m_numRegressions = 0;

    // build logistic model
    if (this.m_numInstances >= m_numFoldsBoosting) {
      if (this.m_fixedNumIterations > 0) {
        this.performBoosting(this.m_fixedNumIterations);
      } else if (this.getUseAIC()) {
        this.performBoostingInfCriterion();
      } else {
        this.performBoostingCV();
      }
    }

    this.m_numParameters += this.m_numRegressions;

    // store performance of model at this node
    Evaluation eval = new Evaluation(this.m_train);
    eval.evaluateModel(this, this.m_train);
    this.m_numIncorrectModel = eval.incorrect();

    boolean grow;
    // split node if more than minNumInstances...
    if (this.m_numInstances > this.m_minNumInstances) {
      // split node: either splitting on class value (a la C4.5) or splitting on
      // residuals
      if (this.m_modelSelection instanceof ResidualModelSelection) {
        // need ps/Ys/Zs/weights
        double[][] probs = this.getProbs(this.getFs(this.m_numericData));
        double[][] trainYs = this.getYs(this.m_train);
        double[][] dataZs = this.getZs(probs, trainYs);
        double[][] dataWs = this.getWs(probs, trainYs);
        this.m_localModel = ((ResidualModelSelection) this.m_modelSelection).selectModel(this.m_train, dataZs, dataWs);
      } else {
        this.m_localModel = this.m_modelSelection.selectModel(this.m_train);
      }
      // ... and valid split found
      grow = (this.m_localModel.numSubsets() > 1);
    } else {
      grow = false;
    }

    if (grow) {
      // create and build children of node
      this.m_isLeaf = false;
      Instances[] localInstances = this.m_localModel.split(this.m_train);

      // don't need data anymore, so clean up
      this.cleanup();

      this.m_sons = new LMTNode[this.m_localModel.numSubsets()];
      for (int i = 0; i < this.m_sons.length; i++) {
        this.m_sons[i] = new LMTNode(this.m_modelSelection, this.m_fixedNumIterations, this.m_fastRegression, this.m_errorOnProbabilities, this.m_minNumInstances,
            this.getWeightTrimBeta(), this.getUseAIC(), this.m_nominalToBinary, this.m_numDecimalPlaces);
        this.m_sons[i].buildTree(localInstances[i], this.copyRegressions(this.m_regressions), this.m_totalInstanceWeight, this.m_numParameters, this.m_numericDataHeader);
        localInstances[i] = null;
      }
    } else {
      this.cleanup();
    }
  }

  /**
   * Prunes a logistic model tree using the CART pruning scheme, given a cost-complexity parameter
   * alpha.
   *
   * @param alpha
   *          the cost-complexity measure
   * @throws Exception
   *           if something goes wrong
   */
  public void prune(final double alpha) throws Exception {

    Vector<LMTNode> nodeList;
    CompareNode comparator = new CompareNode();

    // determine training error of logistic models and subtrees, and calculate
    // alpha-values from them
    this.treeErrors();
    this.calculateAlphas();

    // get list of all inner nodes in the tree
    nodeList = this.getNodes();

    boolean prune = (nodeList.size() > 0);

    while (prune) {

      // select node with minimum alpha
      LMTNode nodeToPrune = Collections.min(nodeList, comparator);

      // want to prune if its alpha is smaller than alpha
      if (nodeToPrune.m_alpha > alpha) {
        break;
      }

      nodeToPrune.m_isLeaf = true;
      nodeToPrune.m_sons = null;

      // update tree errors and alphas
      this.treeErrors();
      this.calculateAlphas();

      nodeList = this.getNodes();
      prune = (nodeList.size() > 0);
    }

    // discard references to models at internal nodes because they are not
    // needed
    for (Object node : this.getNodes()) {
      LMTNode lnode = (LMTNode) node;
      if (!lnode.m_isLeaf) {
        this.m_regressions = null;
      }
    }
  }

  /**
   * Method for performing one fold in the cross-validation of the cost-complexity parameter.
   * Generates a sequence of alpha-values with error estimates for the corresponding (partially
   * pruned) trees, given the test set of that fold.
   *
   * @param alphas
   *          array to hold the generated alpha-values
   * @param errors
   *          array to hold the corresponding error estimates
   * @param test
   *          test set of that fold (to obtain error estimates)
   * @throws Exception
   *           if something goes wrong
   */
  public int prune(final double[] alphas, final double[] errors, final Instances test) throws Exception {

    Vector<LMTNode> nodeList;

    CompareNode comparator = new CompareNode();

    // determine training error of logistic models and subtrees, and calculate
    // alpha-values from them
    this.treeErrors();
    this.calculateAlphas();

    // get list of all inner nodes in the tree
    nodeList = this.getNodes();

    boolean prune = (nodeList.size() > 0);

    // alpha_0 is always zero (unpruned tree)
    alphas[0] = 0;

    Evaluation eval;

    // error of unpruned tree
    if (errors != null) {
      eval = new Evaluation(test);
      eval.evaluateModel(this, test);
      errors[0] = eval.errorRate();
    }

    int iteration = 0;
    while (prune) {

      iteration++;

      // get node with minimum alpha
      LMTNode nodeToPrune = Collections.min(nodeList, comparator);

      nodeToPrune.m_isLeaf = true;
      // Do not set m_sons null, want to unprune

      // get alpha-value of node
      alphas[iteration] = nodeToPrune.m_alpha;

      // log error
      if (errors != null) {
        eval = new Evaluation(test);
        eval.evaluateModel(this, test);
        errors[iteration] = eval.errorRate();
      }

      // update errors/alphas
      this.treeErrors();
      this.calculateAlphas();

      nodeList = this.getNodes();
      prune = (nodeList.size() > 0);
    }

    // set last alpha 1 to indicate end
    alphas[iteration + 1] = 1.0;
    return iteration;
  }

  /**
   * Method to "unprune" a logistic model tree. Sets all leaf-fields to false. Faster than re-growing
   * the tree because the logistic models do not have to be fit again.
   */
  protected void unprune() {
    if (this.m_sons != null) {
      this.m_isLeaf = false;
      for (LMTNode m_son : this.m_sons) {
        m_son.unprune();
      }
    }
  }

  /**
   * Determines the optimum number of LogitBoost iterations to perform by building a standalone
   * logistic regression function on the training data. Used for the heuristic that avoids
   * cross-validating this number again at every node.
   *
   * @param data
   *          training instances for the logistic model
   * @throws Exception
   *           if something goes wrong
   */
  protected int tryLogistic(final Instances data) throws Exception {

    // convert nominal attributes
    Instances filteredData = Filter.useFilter(data, this.m_nominalToBinary);

    LogisticBase logistic = new LogisticBase(0, true, this.m_errorOnProbabilities);

    // limit LogitBoost to 200 iterations (speed)
    logistic.setMaxIterations(200);
    logistic.setWeightTrimBeta(this.getWeightTrimBeta()); // Not in Marc's code.
    // Added by Eibe.
    logistic.setUseAIC(this.getUseAIC());
    logistic.buildClassifier(filteredData);

    // return best number of iterations
    return logistic.getNumRegressions();
  }

  /**
   * Method to count the number of inner nodes in the tree
   *
   * @return the number of inner nodes
   */
  public int getNumInnerNodes() {
    if (this.m_isLeaf) {
      return 0;
    }
    int numNodes = 1;
    for (LMTNode m_son : this.m_sons) {
      numNodes += m_son.getNumInnerNodes();
    }
    return numNodes;
  }

  /**
   * Returns the number of leaves in the tree. Leaves are only counted if their logistic model has
   * changed compared to the one of the parent node.
   *
   * @return the number of leaves
   */
  public int getNumLeaves() {
    int numLeaves;
    if (!this.m_isLeaf) {
      numLeaves = 0;
      int numEmptyLeaves = 0;
      for (int i = 0; i < this.m_sons.length; i++) {
        numLeaves += this.m_sons[i].getNumLeaves();
        if (this.m_sons[i].m_isLeaf && !this.m_sons[i].hasModels()) {
          numEmptyLeaves++;
        }
      }
      if (numEmptyLeaves > 1) {
        numLeaves -= (numEmptyLeaves - 1);
      }
    } else {
      numLeaves = 1;
    }
    return numLeaves;
  }

  /**
   * Updates the numIncorrectTree field for all nodes. This is needed for calculating the
   * alpha-values.
   */
  public void treeErrors() {
    if (this.m_isLeaf) {
      this.m_numIncorrectTree = this.m_numIncorrectModel;
    } else {
      this.m_numIncorrectTree = 0;
      for (LMTNode m_son : this.m_sons) {
        m_son.treeErrors();
        this.m_numIncorrectTree += m_son.m_numIncorrectTree;
      }
    }
  }

  /**
   * Updates the alpha field for all nodes.
   */
  public void calculateAlphas() throws Exception {

    if (!this.m_isLeaf) {
      double errorDiff = this.m_numIncorrectModel - this.m_numIncorrectTree;

      if (errorDiff <= 0) {
        // split increases training error (should not normally happen).
        // prune it instantly.
        this.m_isLeaf = true;
        this.m_sons = null;
        this.m_alpha = Double.MAX_VALUE;
      } else {
        // compute alpha
        errorDiff /= this.m_totalInstanceWeight;
        this.m_alpha = errorDiff / (this.getNumLeaves() - 1);

        for (LMTNode m_son : this.m_sons) {
          m_son.calculateAlphas();
        }
      }
    } else {
      // alpha = infinite for leaves (do not want to prune)
      this.m_alpha = Double.MAX_VALUE;
    }
  }

  /**
   * Return a list of all inner nodes in the tree
   *
   * @return the list of nodes
   */
  public Vector<LMTNode> getNodes() {
    Vector<LMTNode> nodeList = new Vector<>();
    this.getNodes(nodeList);
    return nodeList;
  }

  /**
   * Fills a list with all inner nodes in the tree
   *
   * @param nodeList
   *          the list to be filled
   */
  public void getNodes(final Vector<LMTNode> nodeList) {
    if (!this.m_isLeaf) {
      nodeList.add(this);
      for (LMTNode m_son : this.m_sons) {
        m_son.getNodes(nodeList);
      }
    }
  }

  /**
   * Returns a numeric version of a set of instances. All nominal attributes are replaced by binary
   * ones, and the class variable is replaced by a pseudo-class variable that is used by LogitBoost.
   */
  @Override
  protected Instances getNumericData(final Instances train) throws Exception {

    Instances filteredData = Filter.useFilter(train, this.m_nominalToBinary);

    return super.getNumericData(filteredData);
  }

  /**
   * Returns true if the logistic regression model at this node has changed compared to the one at the
   * parent node.
   *
   * @return whether it has changed
   */
  public boolean hasModels() {
    return (this.m_numRegressions > 0);
  }

  /**
   * Returns the class probabilities for an instance according to the logistic model at the node.
   *
   * @param instance
   *          the instance
   * @return the array of probabilities
   */
  public double[] modelDistributionForInstance(Instance instance) throws Exception {

    // make copy and convert nominal attributes
    this.m_nominalToBinary.input(instance);
    instance = this.m_nominalToBinary.output();

    // saet numeric pseudo-class
    instance.setDataset(this.m_numericDataHeader);

    return this.probs(this.getFs(instance));
  }

  /**
   * Returns the class probabilities for an instance given by the logistic model tree.
   *
   * @param instance
   *          the instance
   * @return the array of probabilities
   */
  @Override
  public double[] distributionForInstance(final Instance instance) throws Exception {

    double[] probs;

    if (this.m_isLeaf) {
      // leaf: use logistic model
      probs = this.modelDistributionForInstance(instance);
    } else {
      // sort into appropiate child node
      int branch = this.m_localModel.whichSubset(instance);
      probs = this.m_sons[branch].distributionForInstance(instance);
    }
    return probs;
  }

  /**
   * Returns the number of leaves (normal count).
   *
   * @return the number of leaves
   */
  public int numLeaves() {
    if (this.m_isLeaf) {
      return 1;
    }
    int numLeaves = 0;
    for (LMTNode m_son : this.m_sons) {
      numLeaves += m_son.numLeaves();
    }
    return numLeaves;
  }

  /**
   * Returns the number of nodes.
   *
   * @return the number of nodes
   */
  public int numNodes() {
    if (this.m_isLeaf) {
      return 1;
    }
    int numNodes = 1;
    for (LMTNode m_son : this.m_sons) {
      numNodes += m_son.numNodes();
    }
    return numNodes;
  }

  /**
   * Returns a description of the logistic model tree (tree structure and logistic models)
   *
   * @return describing string
   */
  @Override
  public String toString() {
    // assign numbers to logistic regression functions at leaves
    this.assignLeafModelNumbers(0);
    try {
      StringBuffer text = new StringBuffer();

      if (this.m_isLeaf) {
        text.append(": ");
        text.append("LM_" + this.m_leafModelNum + ":" + this.getModelParameters());
      } else {
        this.dumpTree(0, text);
      }
      text.append("\n\nNumber of Leaves  : \t" + this.numLeaves() + "\n");
      text.append("\nSize of the Tree : \t" + this.numNodes() + "\n");

      // This prints logistic models after the tree, comment out if only tree
      // should be printed
      text.append(this.modelsToString());
      return text.toString();
    } catch (Exception e) {
      return "Can't print logistic model tree";
    }

  }

  /**
   * Returns a string describing the number of LogitBoost iterations performed at this node, the total
   * number of LogitBoost iterations performed (including iterations at higher levels in the tree),
   * and the number of training examples at this node.
   *
   * @return the describing string
   */
  public String getModelParameters() {

    StringBuffer text = new StringBuffer();
    int numModels = (int) this.m_numParameters;
    text.append(this.m_numRegressions + "/" + numModels + " (" + this.m_numInstances + ")");
    return text.toString();
  }

  /**
   * Help method for printing tree structure.
   *
   * @throws Exception
   *           if something goes wrong
   */
  protected void dumpTree(final int depth, final StringBuffer text) throws Exception {

    for (int i = 0; i < this.m_sons.length; i++) {
      text.append("\n");
      for (int j = 0; j < depth; j++) {
        text.append("|   ");
      }
      text.append(this.m_localModel.leftSide(this.m_train));
      text.append(this.m_localModel.rightSide(i, this.m_train));
      if (this.m_sons[i].m_isLeaf) {
        text.append(": ");
        text.append("LM_" + this.m_sons[i].m_leafModelNum + ":" + this.m_sons[i].getModelParameters());
      } else {
        this.m_sons[i].dumpTree(depth + 1, text);
      }
    }
  }

  /**
   * Assigns unique IDs to all nodes in the tree
   */
  public int assignIDs(final int lastID) {

    int currLastID = lastID + 1;

    this.m_id = currLastID;
    if (this.m_sons != null) {
      for (LMTNode m_son : this.m_sons) {
        currLastID = m_son.assignIDs(currLastID);
      }
    }
    return currLastID;
  }

  /**
   * Assigns numbers to the logistic regression models at the leaves of the tree
   */
  public int assignLeafModelNumbers(int leafCounter) {
    if (!this.m_isLeaf) {
      this.m_leafModelNum = 0;
      for (LMTNode m_son : this.m_sons) {
        leafCounter = m_son.assignLeafModelNumbers(leafCounter);
      }
    } else {
      leafCounter++;
      this.m_leafModelNum = leafCounter;
    }
    return leafCounter;
  }

  /**
   * Returns a string describing the logistic regression function at the node.
   */
  public String modelsToString() {

    StringBuffer text = new StringBuffer();
    if (this.m_isLeaf) {
      text.append("LM_" + this.m_leafModelNum + ":" + super.toString());
    } else {
      for (LMTNode m_son : this.m_sons) {
        text.append("\n" + m_son.modelsToString());
      }
    }
    return text.toString();
  }

  /**
   * Returns graph describing the tree.
   *
   * @throws Exception
   *           if something goes wrong
   */
  public String graph() throws Exception {

    StringBuffer text = new StringBuffer();

    this.assignIDs(-1);
    this.assignLeafModelNumbers(0);
    text.append("digraph LMTree {\n");
    if (this.m_isLeaf) {
      text.append("N" + this.m_id + " [label=\"LM_" + this.m_leafModelNum + ":" + this.getModelParameters() + "\" " + "shape=box style=filled");
      text.append("]\n");
    } else {
      text.append("N" + this.m_id + " [label=\"" + Utils.backQuoteChars(this.m_localModel.leftSide(this.m_train)) + "\" ");
      text.append("]\n");
      this.graphTree(text);
    }

    return text.toString() + "}\n";
  }

  /**
   * Helper function for graph description of tree
   *
   * @throws Exception
   *           if something goes wrong
   */
  private void graphTree(final StringBuffer text) throws Exception {

    for (int i = 0; i < this.m_sons.length; i++) {
      text.append("N" + this.m_id + "->" + "N" + this.m_sons[i].m_id + " [label=\"" + Utils.backQuoteChars(this.m_localModel.rightSide(i, this.m_train).trim()) + "\"]\n");
      if (this.m_sons[i].m_isLeaf) {
        text.append("N" + this.m_sons[i].m_id + " [label=\"LM_" + this.m_sons[i].m_leafModelNum + ":" + this.m_sons[i].getModelParameters() + "\" " + "shape=box style=filled");
        text.append("]\n");
      } else {
        text.append("N" + this.m_sons[i].m_id + " [label=\"" + Utils.backQuoteChars(this.m_sons[i].m_localModel.leftSide(this.m_train)) + "\" ");
        text.append("]\n");
        this.m_sons[i].graphTree(text);
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
