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

/**
 *    KStarNominalAttribute.java
 *    Copyright (C) 1995-2012 Univeristy of Waikato
 *    Java port to Weka by Abdelaziz Mahoui (am14@cs.waikato.ac.nz).
 *
 */

package weka.classifiers.lazy.kstar;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;

/**
 * A custom class which provides the environment for computing the transformation probability of a specified test instance nominal attribute to a specified train instance nominal attribute.
 *
 * @author Len Trigg (len@reeltwo.com)
 * @author Abdelaziz Mahoui (am14@cs.waikato.ac.nz)
 * @version $Revision 1.0 $
 */
public class KStarNominalAttribute implements KStarConstants, RevisionHandler {

	/** The training instances used for classification. */
	protected Instances m_TrainSet;

	/** The test instance */
	protected Instance m_Test;

	/** The train instance */
	protected Instance m_Train;

	/** The index of the nominal attribute in the test and train instances */
	protected int m_AttrIndex;

	/** The stop parameter */
	protected double m_Stop = 1.0;

	/**
	 * Probability of test attribute transforming into train attribute with missing value
	 */
	protected double m_MissingProb = 1.0;

	/**
	 * Average probability of test attribute transforming into train attribute
	 */
	protected double m_AverageProb = 1.0;

	/**
	 * Smallest probability of test attribute transforming into train attribute
	 */
	protected double m_SmallestProb = 1.0;

	/** Number of trai instances with no missing attribute values */
	protected int m_TotalCount;

	/** Distribution of the attribute value in the train dataset */
	protected int[] m_Distribution;

	/**
	 * Set of colomns: each colomn representing a randomised version of the train dataset class colomn
	 */
	protected int[][] m_RandClassCols;

	/**
	 * A cache for storing attribute values and their corresponding stop parameters
	 */
	protected KStarCache m_Cache;

	// KStar Global settings

	/** The number of instances in the dataset */
	protected int m_NumInstances;

	/** The number of class values */
	protected int m_NumClasses;

	/** The number of attributes */
	protected int m_NumAttributes;

	/** The class attribute type */
	protected int m_ClassType;

	/** missing value treatment */
	protected int m_MissingMode = M_AVERAGE;

	/** B_SPHERE = use specified blend, B_ENTROPY = entropic blend setting */
	protected int m_BlendMethod = B_SPHERE;

	/** default sphere of influence blend setting */
	protected int m_BlendFactor = 20;

	/**
	 * Constructor
	 */
	public KStarNominalAttribute(final Instance test, final Instance train, final int attrIndex, final Instances trainSet, final int[][] randClassCol, final KStarCache cache) {
		this.m_Test = test;
		this.m_Train = train;
		this.m_AttrIndex = attrIndex;
		this.m_TrainSet = trainSet;
		this.m_RandClassCols = randClassCol;
		this.m_Cache = cache;
		this.init();
	}

	/**
	 * Initializes the m_Attributes of the class.
	 */
	private void init() {
		try {
			this.m_NumInstances = this.m_TrainSet.numInstances();
			this.m_NumClasses = this.m_TrainSet.numClasses();
			this.m_NumAttributes = this.m_TrainSet.numAttributes();
			this.m_ClassType = this.m_TrainSet.classAttribute().type();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Calculates the probability of the indexed nominal attribute of the test instance transforming into the indexed nominal attribute of the training instance.
	 *
	 * @return the value of the transformation probability.
	 */
	public double transProb() throws InterruptedException {
		double transProb = 0.0;
		// check if the attribute value has been encountred before
		// in which case it should be in the nominal cache
		if (this.m_Cache.containsKey(this.m_Test.value(this.m_AttrIndex))) {
			KStarCache.TableEntry te = this.m_Cache.getCacheValues(this.m_Test.value(this.m_AttrIndex));
			this.m_Stop = te.value;
			this.m_MissingProb = te.pmiss;
		} else {
			this.generateAttrDistribution();
			// we have to compute the parameters
			if (this.m_BlendMethod == B_ENTROPY) {
				this.m_Stop = this.stopProbUsingEntropy();
			} else { // default is B_SPHERE
				this.m_Stop = this.stopProbUsingBlend();
			}
			// store the values in cache
			this.m_Cache.store(this.m_Test.value(this.m_AttrIndex), this.m_Stop, this.m_MissingProb);
		}
		// we've got our m_Stop, then what?
		if (this.m_Train.isMissing(this.m_AttrIndex)) {
			transProb = this.m_MissingProb;
		} else {
			try {
				transProb = (1.0 - this.m_Stop) / this.m_Test.attribute(this.m_AttrIndex).numValues();
				if ((int) this.m_Test.value(this.m_AttrIndex) == (int) this.m_Train.value(this.m_AttrIndex)) {
					transProb += this.m_Stop;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return transProb;
	}

	/**
	 * Calculates the "stop parameter" for this attribute using the entropy method: the value is computed using a root finder algorithm. The method takes advantage of the calculation to compute the smallest and average transformation
	 * probabilities once the stop factor is obtained. It also sets the transformation probability to an attribute with a missing value.
	 *
	 * @return the value of the stop parameter.
	 *
	 */
	private double stopProbUsingEntropy() {
		String debug = "(KStarNominalAttribute.stopProbUsingEntropy)";
		if (this.m_ClassType != Attribute.NOMINAL) {
			System.err.println("Error: " + debug + " attribute class must be nominal!");
			System.exit(1);
		}
		int itcount = 0;
		double stopProb;
		double lower, upper, pstop;
		double bestminprob = 0.0, bestpsum = 0.0;
		double bestdiff = 0.0, bestpstop = 0.0;
		double currentdiff, lastdiff, stepsize, delta;

		KStarWrapper botvals = new KStarWrapper();
		KStarWrapper upvals = new KStarWrapper();
		KStarWrapper vals = new KStarWrapper();

		// Initial values for root finder
		lower = 0.0 + ROOT_FINDER_ACCURACY / 2.0;
		upper = 1.0 - ROOT_FINDER_ACCURACY / 2.0;

		// Find (approx) entropy ranges
		this.calculateEntropy(upper, upvals);
		this.calculateEntropy(lower, botvals);

		if (upvals.avgProb == 0) {
			// When there are no training instances with the test value:
			// doesn't matter what exact value we use for pstop, just acts as
			// a constant scale factor in this case.
			this.calculateEntropy(lower, vals);
		} else {
			// Optimise the scale factor
			if ((upvals.randEntropy - upvals.actEntropy < botvals.randEntropy - botvals.actEntropy) && (botvals.randEntropy - botvals.actEntropy > FLOOR)) {
				bestpstop = pstop = lower;
				stepsize = INITIAL_STEP;
				bestminprob = botvals.minProb;
				bestpsum = botvals.avgProb;
			} else {
				bestpstop = pstop = upper;
				stepsize = -INITIAL_STEP;
				bestminprob = upvals.minProb;
				bestpsum = upvals.avgProb;
			}
			bestdiff = currentdiff = FLOOR;
			itcount = 0;
			/* Enter the root finder */
			while (true) {
				itcount++;
				lastdiff = currentdiff;
				pstop += stepsize;
				if (pstop <= lower) {
					pstop = lower;
					currentdiff = 0.0;
					delta = -1.0;
				} else if (pstop >= upper) {
					pstop = upper;
					currentdiff = 0.0;
					delta = -1.0;
				} else {
					this.calculateEntropy(pstop, vals);
					currentdiff = vals.randEntropy - vals.actEntropy;

					if (currentdiff < FLOOR) {
						currentdiff = FLOOR;
						if ((Math.abs(stepsize) < INITIAL_STEP) && (bestdiff == FLOOR)) {
							bestpstop = lower;
							bestminprob = botvals.minProb;
							bestpsum = botvals.avgProb;
							break;
						}
					}
					delta = currentdiff - lastdiff;
				}
				if (currentdiff > bestdiff) {
					bestdiff = currentdiff;
					bestpstop = pstop;
					bestminprob = vals.minProb;
					bestpsum = vals.avgProb;
				}
				if (delta < 0) {
					if (Math.abs(stepsize) < ROOT_FINDER_ACCURACY) {
						break;
					} else {
						stepsize /= -2.0;
					}
				}
				if (itcount > ROOT_FINDER_MAX_ITER) {
					break;
				}
			}
		}

		this.m_SmallestProb = bestminprob;
		this.m_AverageProb = bestpsum;
		// Set the probability of transforming to a missing value
		switch (this.m_MissingMode) {
		case M_DELETE:
			this.m_MissingProb = 0.0;
			break;
		case M_NORMAL:
			this.m_MissingProb = 1.0;
			break;
		case M_MAXDIFF:
			this.m_MissingProb = this.m_SmallestProb;
			break;
		case M_AVERAGE:
			this.m_MissingProb = this.m_AverageProb;
			break;
		}

		if (Math.abs(bestpsum - this.m_TotalCount) < EPSILON) {
			// No difference in the values
			stopProb = 1.0;
		} else {
			stopProb = bestpstop;
		}
		return stopProb;
	}

	/**
	 * Calculates the entropy of the actual class prediction and the entropy for random class prediction. It also calculates the smallest and average transformation probabilities.
	 *
	 * @param stop
	 *            the stop parameter
	 * @param params
	 *            the object wrapper for the parameters: actual entropy, random entropy, average probability and smallest probability.
	 * @return the values are returned in the object "params".
	 *
	 */
	private void calculateEntropy(final double stop, final KStarWrapper params) {
		int i, j, k;
		Instance train;
		double actent = 0.0, randent = 0.0;
		double pstar, tprob, psum = 0.0, minprob = 1.0;
		double actClassProb, randClassProb;
		double[][] pseudoClassProb = new double[NUM_RAND_COLS + 1][this.m_NumClasses];
		// init ...
		for (j = 0; j <= NUM_RAND_COLS; j++) {
			for (i = 0; i < this.m_NumClasses; i++) {
				pseudoClassProb[j][i] = 0.0;
			}
		}
		for (i = 0; i < this.m_NumInstances; i++) {
			train = this.m_TrainSet.instance(i);
			if (!train.isMissing(this.m_AttrIndex)) {
				pstar = this.PStar(this.m_Test, train, this.m_AttrIndex, stop);
				tprob = pstar / this.m_TotalCount;
				if (pstar < minprob) {
					minprob = pstar;
				}
				psum += tprob;
				// filter instances with same class value
				for (k = 0; k <= NUM_RAND_COLS; k++) {
					// instance i is assigned a random class value in colomn k;
					// colomn k = NUM_RAND_COLS contains the original mapping:
					// instance -> class vlaue
					pseudoClassProb[k][this.m_RandClassCols[k][i]] += tprob;
				}
			}
		}
		// compute the actual entropy using the class probs
		// with the original class value mapping (colomn NUM_RAND_COLS)
		for (j = this.m_NumClasses - 1; j >= 0; j--) {
			actClassProb = pseudoClassProb[NUM_RAND_COLS][j] / psum;
			if (actClassProb > 0) {
				actent -= actClassProb * Math.log(actClassProb) / LOG2;
			}
		}
		// compute a random entropy using the pseudo class probs
		// excluding the colomn NUM_RAND_COLS
		for (k = 0; k < NUM_RAND_COLS; k++) {
			for (i = this.m_NumClasses - 1; i >= 0; i--) {
				randClassProb = pseudoClassProb[k][i] / psum;
				if (randClassProb > 0) {
					randent -= randClassProb * Math.log(randClassProb) / LOG2;
				}
			}
		}
		randent /= NUM_RAND_COLS;
		// return the results ... Yuk !!!
		params.actEntropy = actent;
		params.randEntropy = randent;
		params.avgProb = psum;
		params.minProb = minprob;
	}

	/**
	 * Calculates the "stop parameter" for this attribute using the blend method: the value is computed using a root finder algorithm. The method takes advantage of this calculation to compute the smallest and average transformation
	 * probabilities once the stop factor is obtained. It also sets the transformation probability to an attribute with a missing value.
	 *
	 * @return the value of the stop parameter.
	 *
	 */
	private double stopProbUsingBlend() {
		int itcount = 0;
		double stopProb, aimfor;
		double lower, upper, tstop;

		KStarWrapper botvals = new KStarWrapper();
		KStarWrapper upvals = new KStarWrapper();
		KStarWrapper vals = new KStarWrapper();

		int testvalue = (int) this.m_Test.value(this.m_AttrIndex);
		aimfor = (this.m_TotalCount - this.m_Distribution[testvalue]) * (double) this.m_BlendFactor / 100.0 + this.m_Distribution[testvalue];

		// Initial values for root finder
		tstop = 1.0 - this.m_BlendFactor / 100.0;
		lower = 0.0 + ROOT_FINDER_ACCURACY / 2.0;
		upper = 1.0 - ROOT_FINDER_ACCURACY / 2.0;

		// Find out function border values
		this.calculateSphereSize(testvalue, lower, botvals);
		botvals.sphere -= aimfor;
		this.calculateSphereSize(testvalue, upper, upvals);
		upvals.sphere -= aimfor;

		if (upvals.avgProb == 0) {
			// When there are no training instances with the test value:
			// doesn't matter what exact value we use for tstop, just acts as
			// a constant scale factor in this case.
			this.calculateSphereSize(testvalue, tstop, vals);
		} else if (upvals.sphere > 0) {
			// Can't include aimfor instances, going for min possible
			tstop = upper;
			vals.avgProb = upvals.avgProb;
		} else {
			// Enter the root finder
			for (;;) {
				itcount++;
				this.calculateSphereSize(testvalue, tstop, vals);
				vals.sphere -= aimfor;
				if (Math.abs(vals.sphere) <= ROOT_FINDER_ACCURACY || itcount >= ROOT_FINDER_MAX_ITER) {
					break;
				}
				if (vals.sphere > 0.0) {
					lower = tstop;
					tstop = (upper + lower) / 2.0;
				} else {
					upper = tstop;
					tstop = (upper + lower) / 2.0;
				}
			}
		}

		this.m_SmallestProb = vals.minProb;
		this.m_AverageProb = vals.avgProb;
		// Set the probability of transforming to a missing value
		switch (this.m_MissingMode) {
		case M_DELETE:
			this.m_MissingProb = 0.0;
			break;
		case M_NORMAL:
			this.m_MissingProb = 1.0;
			break;
		case M_MAXDIFF:
			this.m_MissingProb = this.m_SmallestProb;
			break;
		case M_AVERAGE:
			this.m_MissingProb = this.m_AverageProb;
			break;
		}

		if (Math.abs(vals.avgProb - this.m_TotalCount) < EPSILON) {
			// No difference in the values
			stopProb = 1.0;
		} else {
			stopProb = tstop;
		}
		return stopProb;
	}

	/**
	 * Calculates the size of the "sphere of influence" defined as: sphere = sum(P^2)/sum(P)^2 P(i|j) = (1-tstop)*P(i) + ((i==j)?tstop:0). This method takes advantage of the calculation to compute the values of the "smallest" and "average"
	 * transformation probabilities when using the specified stop parameter.
	 *
	 * @param testValue
	 *            the value of the test instance
	 * @param stop
	 *            the stop parameter
	 * @param params
	 *            a wrapper of the parameters to be computed: "sphere" the sphere size "avgprob" the average transformation probability "minProb" the smallest transformation probability
	 * @return the values are returned in "params" object.
	 *
	 */
	private void calculateSphereSize(final int testvalue, final double stop, final KStarWrapper params) {
		int i, thiscount;
		double tprob, tval = 0.0, t1 = 0.0;
		double sphere, minprob = 1.0, transprob = 0.0;

		for (i = 0; i < this.m_Distribution.length; i++) {
			thiscount = this.m_Distribution[i];
			if (thiscount != 0) {
				if (testvalue == i) {
					tprob = (stop + (1 - stop) / this.m_Distribution.length) / this.m_TotalCount;
					tval += tprob * thiscount;
					t1 += tprob * tprob * thiscount;
				} else {
					tprob = ((1 - stop) / this.m_Distribution.length) / this.m_TotalCount;
					tval += tprob * thiscount;
					t1 += tprob * tprob * thiscount;
				}
				if (minprob > tprob * this.m_TotalCount) {
					minprob = tprob * this.m_TotalCount;
				}
			}
		}
		transprob = tval;
		sphere = (t1 == 0) ? 0 : ((tval * tval) / t1);
		// return values ... Yck!!!
		params.sphere = sphere;
		params.avgProb = transprob;
		params.minProb = minprob;
	}

	/**
	 * Calculates the nominal probability function defined as: P(i|j) = (1-stop) * P(i) + ((i==j) ? stop : 0) In this case, it calculates the transformation probability of the indexed test attribute to the indexed train attribute.
	 *
	 * @param test
	 *            the test instance
	 * @param train
	 *            the train instance
	 * @param col
	 *            the attribute index
	 * @return the value of the tranformation probability.
	 *
	 */
	private double PStar(final Instance test, final Instance train, final int col, final double stop) {
		double pstar;
		int numvalues = 0;
		try {
			numvalues = test.attribute(col).numValues();
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		if ((int) test.value(col) == (int) train.value(col)) {
			pstar = stop + (1 - stop) / numvalues;
		} else {
			pstar = (1 - stop) / numvalues;
		}
		return pstar;
	}

	/**
	 * Calculates the distribution, in the dataset, of the indexed nominal attribute values. It also counts the actual number of training instances that contributed (those with non-missing values) to calculate the distribution.
	 */
	private void generateAttrDistribution() throws InterruptedException {
		this.m_Distribution = new int[this.m_TrainSet.attribute(this.m_AttrIndex).numValues()];
		int i;
		Instance train;
		for (i = 0; i < this.m_NumInstances; i++) {
			// XXX interrupt weka
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			train = this.m_TrainSet.instance(i);
			if (!train.isMissing(this.m_AttrIndex)) {
				this.m_TotalCount++;
				this.m_Distribution[(int) train.value(this.m_AttrIndex)]++;
			}
		}
	}

	/**
	 * Sets the options.
	 *
	 */
	public void setOptions(final int missingmode, final int blendmethod, final int blendfactor) {
		this.m_MissingMode = missingmode;
		this.m_BlendMethod = blendmethod;
		this.m_BlendFactor = blendfactor;
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
} // class
