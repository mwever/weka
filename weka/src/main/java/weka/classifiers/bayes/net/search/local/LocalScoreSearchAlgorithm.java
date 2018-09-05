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
 * LocalScoreSearchAlgorithm.java
 * Copyright (C) 2004-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.bayes.net.search.local;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.ParentSet;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Statistics;
import weka.core.Tag;
import weka.core.Utils;

/**
 * <!-- globalinfo-start --> The ScoreBasedSearchAlgorithm class supports Bayes net structure search algorithms that are based on maximizing scores (as opposed to for example conditional independence based search algorithms).
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
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
 * @author Remco Bouckaert
 * @version $Revision$
 */
public class LocalScoreSearchAlgorithm extends SearchAlgorithm {

	/** for serialization */
	static final long serialVersionUID = 3325995552474190374L;

	/** points to Bayes network for which a structure is searched for **/
	BayesNet m_BayesNet;

	/**
	 * default constructor
	 */
	public LocalScoreSearchAlgorithm() {
	} // c'tor

	/**
	 * constructor
	 *
	 * @param bayesNet
	 *            the network
	 * @param instances
	 *            the data
	 */
	public LocalScoreSearchAlgorithm(final BayesNet bayesNet, final Instances instances) {
		this.m_BayesNet = bayesNet;
		// m_Instances = instances;
	} // c'tor

	/**
	 * Holds prior on count
	 */
	double m_fAlpha = 0.5;

	/** the score types */
	public static final Tag[] TAGS_SCORE_TYPE = { new Tag(Scoreable.BAYES, "BAYES"), new Tag(Scoreable.BDeu, "BDeu"), new Tag(Scoreable.MDL, "MDL"), new Tag(Scoreable.ENTROPY, "ENTROPY"), new Tag(Scoreable.AIC, "AIC") };

	/**
	 * Holds the score type used to measure quality of network
	 */
	int m_nScoreType = Scoreable.BAYES;

	/**
	 * logScore returns the log of the quality of a network (e.g. the posterior probability of the network, or the MDL value).
	 *
	 * @param nType
	 *            score type (Bayes, MDL, etc) to calculate score with
	 * @return log score.
	 */
	public double logScore(int nType) {
		if (this.m_BayesNet.m_Distributions == null) {
			return 0;
		}
		if (nType < 0) {
			nType = this.m_nScoreType;
		}

		double fLogScore = 0.0;

		Instances instances = this.m_BayesNet.m_Instances;

		for (int iAttribute = 0; iAttribute < instances.numAttributes(); iAttribute++) {
			int nCardinality = this.m_BayesNet.getParentSet(iAttribute).getCardinalityOfParents();
			for (int iParent = 0; iParent < nCardinality; iParent++) {
				fLogScore += ((Scoreable) this.m_BayesNet.m_Distributions[iAttribute][iParent]).logScore(nType, nCardinality);
			}

			switch (nType) {
			case (Scoreable.MDL): {
				fLogScore -= 0.5 * this.m_BayesNet.getParentSet(iAttribute).getCardinalityOfParents() * (instances.attribute(iAttribute).numValues() - 1) * Math.log(this.m_BayesNet.getNumInstances());
			}
				break;
			case (Scoreable.AIC): {
				fLogScore -= this.m_BayesNet.getParentSet(iAttribute).getCardinalityOfParents() * (instances.attribute(iAttribute).numValues() - 1);
			}
				break;
			}
		}

		return fLogScore;
	} // logScore

	/**
	 * buildStructure determines the network structure/graph of the network with the K2 algorithm, restricted by its initial structure (which can be an empty graph, or a Naive Bayes graph.
	 *
	 * @param bayesNet
	 *            the network
	 * @param instances
	 *            the data to use
	 * @throws Exception
	 *             if something goes wrong
	 */
	@Override
	public void buildStructure(final BayesNet bayesNet, final Instances instances) throws Exception {
		this.m_BayesNet = bayesNet;
		super.buildStructure(bayesNet, instances);
	} // buildStructure

	/**
	 * Calc Node Score for given parent set
	 *
	 * @param nNode
	 *            node for which the score is calculate
	 * @return log score
	 * @throws InterruptedException
	 */
	public double calcNodeScore(final int nNode) throws InterruptedException {
		if (this.m_BayesNet.getUseADTree() && this.m_BayesNet.getADTree() != null) {
			return this.calcNodeScoreADTree(nNode);
		} else {
			return this.calcNodeScorePlain(nNode);
		}
	}

	/**
	 * helper function for CalcNodeScore above using the ADTree data structure
	 *
	 * @param nNode
	 *            node for which the score is calculate
	 * @return log score
	 * @throws InterruptedException
	 */
	private double calcNodeScoreADTree(final int nNode) throws InterruptedException {
		Instances instances = this.m_BayesNet.m_Instances;
		ParentSet oParentSet = this.m_BayesNet.getParentSet(nNode);
		// get set of parents, insert iNode
		int nNrOfParents = oParentSet.getNrOfParents();
		int[] nNodes = new int[nNrOfParents + 1];
		for (int iParent = 0; iParent < nNrOfParents; iParent++) {
			nNodes[iParent] = oParentSet.getParent(iParent);
		}
		nNodes[nNrOfParents] = nNode;

		// calculate offsets
		int[] nOffsets = new int[nNrOfParents + 1];
		int nOffset = 1;
		nOffsets[nNrOfParents] = 1;
		nOffset *= instances.attribute(nNode).numValues();
		for (int iNode = nNrOfParents - 1; iNode >= 0; iNode--) {
			nOffsets[iNode] = nOffset;
			nOffset *= instances.attribute(nNodes[iNode]).numValues();
		}

		// sort nNodes & offsets
		for (int iNode = 1; iNode < nNodes.length; iNode++) {
			int iNode2 = iNode;
			while (iNode2 > 0 && nNodes[iNode2] < nNodes[iNode2 - 1]) {
				int h = nNodes[iNode2];
				nNodes[iNode2] = nNodes[iNode2 - 1];
				nNodes[iNode2 - 1] = h;
				h = nOffsets[iNode2];
				nOffsets[iNode2] = nOffsets[iNode2 - 1];
				nOffsets[iNode2 - 1] = h;
				iNode2--;
			}
		}

		// get counts from ADTree
		int nCardinality = oParentSet.getCardinalityOfParents();
		int numValues = instances.attribute(nNode).numValues();
		int[] nCounts = new int[nCardinality * numValues];
		// if (nNrOfParents > 1) {

		this.m_BayesNet.getADTree().getCounts(nCounts, nNodes, nOffsets, 0, 0, false);

		return this.calcScoreOfCounts(nCounts, nCardinality, numValues, instances);
	} // CalcNodeScore

	private double calcNodeScorePlain(final int nNode) throws InterruptedException {
		Instances instances = this.m_BayesNet.m_Instances;
		ParentSet oParentSet = this.m_BayesNet.getParentSet(nNode);

		// determine cardinality of parent set & reserve space for frequency counts
		int nCardinality = oParentSet.getCardinalityOfParents();
		int numValues = instances.attribute(nNode).numValues();
		int[] nCounts = new int[nCardinality * numValues];

		// initialize (don't need this?)
		for (int iParent = 0; iParent < nCardinality * numValues; iParent++) {
			nCounts[iParent] = 0;
		}

		// estimate distributions
		Enumeration<Instance> enumInsts = instances.enumerateInstances();

		while (enumInsts.hasMoreElements()) {
			Instance instance = enumInsts.nextElement();

			// updateClassifier;
			double iCPT = 0;

			for (int iParent = 0; iParent < oParentSet.getNrOfParents(); iParent++) {
				int nParent = oParentSet.getParent(iParent);

				iCPT = iCPT * instances.attribute(nParent).numValues() + instance.value(nParent);
			}

			nCounts[numValues * ((int) iCPT) + (int) instance.value(nNode)]++;
		}

		return this.calcScoreOfCounts(nCounts, nCardinality, numValues, instances);
	} // CalcNodeScore

	/**
	 * utility function used by CalcScore and CalcNodeScore to determine the score based on observed frequencies.
	 *
	 * @param nCounts
	 *            array with observed frequencies
	 * @param nCardinality
	 *            ardinality of parent set
	 * @param numValues
	 *            number of values a node can take
	 * @param instances
	 *            to calc score with
	 * @return log score
	 * @throws InterruptedException
	 */
	protected double calcScoreOfCounts(final int[] nCounts, final int nCardinality, final int numValues, final Instances instances) throws InterruptedException {

		// calculate scores using the distributions
		double fLogScore = 0.0;

		for (int iParent = 0; iParent < nCardinality; iParent++) {
			// XXX kill weka
			if (Thread.currentThread().isInterrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			switch (this.m_nScoreType) {

			case (Scoreable.BAYES): {
				double nSumOfCounts = 0;

				for (int iSymbol = 0; iSymbol < numValues; iSymbol++) {
					// XXX kill weka
					if (Thread.currentThread().isInterrupted()) {
						throw new InterruptedException("Killed WEKA!");
					}
					if (this.m_fAlpha + nCounts[iParent * numValues + iSymbol] != 0) {
						fLogScore += Statistics.lnGamma(this.m_fAlpha + nCounts[iParent * numValues + iSymbol]);
						nSumOfCounts += this.m_fAlpha + nCounts[iParent * numValues + iSymbol];
					}
				}

				if (nSumOfCounts != 0) {
					fLogScore -= Statistics.lnGamma(nSumOfCounts);
				}

				if (this.m_fAlpha != 0) {
					fLogScore -= numValues * Statistics.lnGamma(this.m_fAlpha);
					fLogScore += Statistics.lnGamma(numValues * this.m_fAlpha);
				}
			}

				break;
			case (Scoreable.BDeu): {
				double nSumOfCounts = 0;

				for (int iSymbol = 0; iSymbol < numValues; iSymbol++) {
					// XXX kill weka
					if (Thread.currentThread().isInterrupted()) {
						throw new InterruptedException("Killed WEKA!");
					}
					if (this.m_fAlpha + nCounts[iParent * numValues + iSymbol] != 0) {
						fLogScore += Statistics.lnGamma(1.0 / (numValues * nCardinality) + nCounts[iParent * numValues + iSymbol]);
						nSumOfCounts += 1.0 / (numValues * nCardinality) + nCounts[iParent * numValues + iSymbol];
					}
				}
				fLogScore -= Statistics.lnGamma(nSumOfCounts);

				fLogScore -= numValues * Statistics.lnGamma(1.0 / (numValues * nCardinality));
				fLogScore += Statistics.lnGamma(1.0 / nCardinality);
			}
				break;

			case (Scoreable.MDL):

			case (Scoreable.AIC):

			case (Scoreable.ENTROPY): {
				double nSumOfCounts = 0;

				for (int iSymbol = 0; iSymbol < numValues; iSymbol++) {
					// XXX kill weka
					if (Thread.currentThread().isInterrupted()) {
						throw new InterruptedException("Killed WEKA!");
					}
					nSumOfCounts += nCounts[iParent * numValues + iSymbol];
				}

				for (int iSymbol = 0; iSymbol < numValues; iSymbol++) {
					// XXX kill weka
					if (Thread.currentThread().isInterrupted()) {
						throw new InterruptedException("Killed WEKA!");
					}
					if (nCounts[iParent * numValues + iSymbol] > 0) {
						fLogScore += nCounts[iParent * numValues + iSymbol] * Math.log(nCounts[iParent * numValues + iSymbol] / nSumOfCounts);
					}
				}
			}

				break;

			default: {
			}
			}
		}

		switch (this.m_nScoreType) {

		case (Scoreable.MDL): {
			fLogScore -= 0.5 * nCardinality * (numValues - 1) * Math.log(this.m_BayesNet.getNumInstances());

			// it seems safe to assume that numInstances>0 here
		}

			break;

		case (Scoreable.AIC): {
			fLogScore -= nCardinality * (numValues - 1);
		}

			break;
		}

		return fLogScore;
	} // CalcNodeScore

	protected double calcScoreOfCounts2(final int[][] nCounts, final int nCardinality, final int numValues, final Instances instances) throws InterruptedException {

		// calculate scores using the distributions
		double fLogScore = 0.0;

		for (int iParent = 0; iParent < nCardinality; iParent++) {
			// XXX kill weka
			if (Thread.currentThread().isInterrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			switch (this.m_nScoreType) {

			case (Scoreable.BAYES): {
				double nSumOfCounts = 0;

				for (int iSymbol = 0; iSymbol < numValues; iSymbol++) {
					// XXX kill weka
					if (Thread.currentThread().isInterrupted()) {
						throw new InterruptedException("Killed WEKA!");
					}
					if (this.m_fAlpha + nCounts[iParent][iSymbol] != 0) {
						fLogScore += Statistics.lnGamma(this.m_fAlpha + nCounts[iParent][iSymbol]);
						nSumOfCounts += this.m_fAlpha + nCounts[iParent][iSymbol];
					}
				}

				if (nSumOfCounts != 0) {
					fLogScore -= Statistics.lnGamma(nSumOfCounts);
				}

				if (this.m_fAlpha != 0) {
					fLogScore -= numValues * Statistics.lnGamma(this.m_fAlpha);
					fLogScore += Statistics.lnGamma(numValues * this.m_fAlpha);
				}
			}

				break;

			case (Scoreable.BDeu): {
				double nSumOfCounts = 0;

				for (int iSymbol = 0; iSymbol < numValues; iSymbol++) {
					// XXX kill weka
					if (Thread.currentThread().isInterrupted()) {
						throw new InterruptedException("Killed WEKA!");
					}
					if (this.m_fAlpha + nCounts[iParent][iSymbol] != 0) {
						fLogScore += Statistics.lnGamma(1.0 / (numValues * nCardinality) + nCounts[iParent][iSymbol]);
						nSumOfCounts += 1.0 / (numValues * nCardinality) + nCounts[iParent][iSymbol];
					}
				}
				fLogScore -= Statistics.lnGamma(nSumOfCounts);

				fLogScore -= numValues * Statistics.lnGamma(1.0 / (nCardinality * numValues));
				fLogScore += Statistics.lnGamma(1.0 / nCardinality);
			}
				break;

			case (Scoreable.MDL):

			case (Scoreable.AIC):

			case (Scoreable.ENTROPY): {
				double nSumOfCounts = 0;

				for (int iSymbol = 0; iSymbol < numValues; iSymbol++) {
					// XXX kill weka
					if (Thread.currentThread().isInterrupted()) {
						throw new InterruptedException("Killed WEKA!");
					}
					nSumOfCounts += nCounts[iParent][iSymbol];
				}

				for (int iSymbol = 0; iSymbol < numValues; iSymbol++) {
					// XXX kill weka
					if (Thread.currentThread().isInterrupted()) {
						throw new InterruptedException("Killed WEKA!");
					}
					if (nCounts[iParent][iSymbol] > 0) {
						fLogScore += nCounts[iParent][iSymbol] * Math.log(nCounts[iParent][iSymbol] / nSumOfCounts);
					}
				}
			}

				break;

			default: {
			}
			}
		}

		switch (this.m_nScoreType) {

		case (Scoreable.MDL): {
			fLogScore -= 0.5 * nCardinality * (numValues - 1) * Math.log(this.m_BayesNet.getNumInstances());

			// it seems safe to assume that numInstances>0 here
		}

			break;

		case (Scoreable.AIC): {
			fLogScore -= nCardinality * (numValues - 1);
		}

			break;
		}

		return fLogScore;
	} // CalcNodeScore

	/**
	 * Calc Node Score With AddedParent
	 *
	 * @param nNode
	 *            node for which the score is calculate
	 * @param nCandidateParent
	 *            candidate parent to add to the existing parent set
	 * @return log score
	 * @throws InterruptedException
	 */
	public double calcScoreWithExtraParent(final int nNode, final int nCandidateParent) throws InterruptedException {
		ParentSet oParentSet = this.m_BayesNet.getParentSet(nNode);

		// sanity check: nCandidateParent should not be in parent set already
		if (oParentSet.contains(nCandidateParent)) {
			return -1e100;
		}

		// set up candidate parent
		oParentSet.addParent(nCandidateParent, this.m_BayesNet.m_Instances);

		// calculate the score
		double logScore = this.calcNodeScore(nNode);

		// delete temporarily added parent
		oParentSet.deleteLastParent(this.m_BayesNet.m_Instances);

		return logScore;
	} // CalcScoreWithExtraParent

	/**
	 * Calc Node Score With Parent Deleted
	 *
	 * @param nNode
	 *            node for which the score is calculate
	 * @param nCandidateParent
	 *            candidate parent to delete from the existing parent set
	 * @return log score
	 * @throws InterruptedException
	 */
	public double calcScoreWithMissingParent(final int nNode, final int nCandidateParent) throws InterruptedException {
		ParentSet oParentSet = this.m_BayesNet.getParentSet(nNode);

		// sanity check: nCandidateParent should be in parent set already
		if (!oParentSet.contains(nCandidateParent)) {
			return -1e100;
		}

		// set up candidate parent
		int iParent = oParentSet.deleteParent(nCandidateParent, this.m_BayesNet.m_Instances);

		// calculate the score
		double logScore = this.calcNodeScore(nNode);

		// restore temporarily deleted parent
		oParentSet.addParent(nCandidateParent, iParent, this.m_BayesNet.m_Instances);

		return logScore;
	} // CalcScoreWithMissingParent

	/**
	 * set quality measure to be used in searching for networks.
	 *
	 * @param newScoreType
	 *            the new score type
	 */
	public void setScoreType(final SelectedTag newScoreType) {
		if (newScoreType.getTags() == TAGS_SCORE_TYPE) {
			this.m_nScoreType = newScoreType.getSelectedTag().getID();
		}
	}

	/**
	 * get quality measure to be used in searching for networks.
	 *
	 * @return quality measure
	 */
	public SelectedTag getScoreType() {
		return new SelectedTag(this.m_nScoreType, TAGS_SCORE_TYPE);
	}

	/**
	 *
	 * @param bMarkovBlanketClassifier
	 */
	@Override
	public void setMarkovBlanketClassifier(final boolean bMarkovBlanketClassifier) {
		super.setMarkovBlanketClassifier(bMarkovBlanketClassifier);
	}

	/**
	 *
	 * @return
	 */
	@Override
	public boolean getMarkovBlanketClassifier() {
		return super.getMarkovBlanketClassifier();
	}

	/**
	 * Returns an enumeration describing the available options
	 *
	 * @return an enumeration of all the available options
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>();

		newVector.addElement(new Option("\tApplies a Markov Blanket correction to the network structure, \n" + "\tafter a network structure is learned. This ensures that all \n"
				+ "\tnodes in the network are part of the Markov blanket of the \n" + "\tclassifier node.", "mbc", 0, "-mbc"));

		newVector.addElement(new Option("\tScore type (BAYES, BDeu, MDL, ENTROPY and AIC)", "S", 1, "-S [BAYES|MDL|ENTROPY|AIC|CROSS_CLASSIC|CROSS_BAYES]"));

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

		this.setMarkovBlanketClassifier(Utils.getFlag("mbc", options));

		String sScore = Utils.getOption('S', options);

		if (sScore.compareTo("BAYES") == 0) {
			this.setScoreType(new SelectedTag(Scoreable.BAYES, TAGS_SCORE_TYPE));
		}
		if (sScore.compareTo("BDeu") == 0) {
			this.setScoreType(new SelectedTag(Scoreable.BDeu, TAGS_SCORE_TYPE));
		}
		if (sScore.compareTo("MDL") == 0) {
			this.setScoreType(new SelectedTag(Scoreable.MDL, TAGS_SCORE_TYPE));
		}
		if (sScore.compareTo("ENTROPY") == 0) {
			this.setScoreType(new SelectedTag(Scoreable.ENTROPY, TAGS_SCORE_TYPE));
		}
		if (sScore.compareTo("AIC") == 0) {
			this.setScoreType(new SelectedTag(Scoreable.AIC, TAGS_SCORE_TYPE));
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

		if (this.getMarkovBlanketClassifier()) {
			options.add("-mbc");
		}

		options.add("-S");

		switch (this.m_nScoreType) {

		case (Scoreable.BAYES):
			options.add("BAYES");
			break;

		case (Scoreable.BDeu):
			options.add("BDeu");
			break;

		case (Scoreable.MDL):
			options.add("MDL");
			break;

		case (Scoreable.ENTROPY):
			options.add("ENTROPY");

			break;

		case (Scoreable.AIC):
			options.add("AIC");
			break;
		}

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[0]);
	} // getOptions

	/**
	 * @return a string to describe the ScoreType option.
	 */
	public String scoreTypeTipText() {
		return "The score type determines the measure used to judge the quality of a" + " network structure. It can be one of Bayes, BDeu, Minimum Description Length (MDL)," + " Akaike Information Criterion (AIC), and Entropy.";
	}

	/**
	 * @return a string to describe the MarkovBlanketClassifier option.
	 */
	@Override
	public String markovBlanketClassifierTipText() {
		return super.markovBlanketClassifierTipText();
	}

	/**
	 * This will return a string describing the search algorithm.
	 *
	 * @return The string.
	 */
	public String globalInfo() {
		return "The ScoreBasedSearchAlgorithm class supports Bayes net " + "structure search algorithms that are based on maximizing " + "scores (as opposed to for example conditional independence " + "based search algorithms).";
	} // globalInfo

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
