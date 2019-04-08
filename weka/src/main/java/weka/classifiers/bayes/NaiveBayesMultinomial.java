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
 * NaiveBayesMultinomial.java
 * Copyright (C) 2003-2017 University of Waikato, Hamilton, New Zealand
 */

package weka.classifiers.bayes;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

/**
 * <!-- globalinfo-start --> Class for building and using a multinomial Naive Bayes classifier. For
 * more information see,<br/>
 * <br/>
 * Andrew Mccallum, Kamal Nigam: A Comparison of Event Models for Naive Bayes Text Classification.
 * In: AAAI-98 Workshop on 'Learning for Text Categorization', 1998.<br/>
 * <br/>
 * The core equation for this classifier:<br/>
 * <br/>
 * P[Ci|D] = (P[D|Ci] x P[Ci]) / P[D] (Bayes rule)<br/>
 * <br/>
 * where Ci is class i and D is a document.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 *
 * <pre>
 * &#64;inproceedings{Mccallum1998,
 *    author = {Andrew Mccallum and Kamal Nigam},
 *    booktitle = {AAAI-98 Workshop on 'Learning for Text Categorization'},
 *    title = {A Comparison of Event Models for Naive Bayes Text Classification},
 *    year = {1998}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * -output-debug-info <br>
 * If set, classifier is run in debug mode and may output additional info to the console.
 * <p>
 *
 * -do-not-check-capabilities <br>
 * If set, classifier capabilities are not checked before classifier is built (use with caution).
 * <p>
 *
 * -num-decimal-laces <br>
 * The number of decimal places for the output of numbers in the model.
 * <p>
 *
 * -batch-size <br>
 * The desired batch size for batch prediction.
 * <p>
 *
 * <!-- options-end -->
 *
 * @author Andrew Golightly (acg4@cs.waikato.ac.nz)
 * @author Bernhard Pfahringer (bernhard@cs.waikato.ac.nz)
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class NaiveBayesMultinomial extends AbstractClassifier implements WeightedInstancesHandler, TechnicalInformationHandler {

	/** for serialization */
	static final long serialVersionUID = 5932177440181257085L;

	/**
	 * probability that a word (w) exists in a class (H) (i.e. Pr[w|H]) The matrix is in the this
	 * format: probOfWordGivenClass[class][wordAttribute] NOTE: the values are actually the log of
	 * Pr[w|H]
	 */
	protected double[][] m_probOfWordGivenClass;

	/** the probability of a class (i.e. Pr[H]). */
	protected double[] m_probOfClass;

	/** number of unique words */
	protected int m_numAttributes;

	/** number of class values */
	protected int m_numClasses;

	/** copy of header information for use in toString method */
	protected Instances m_headerInfo;

	/**
	 * Returns a string describing this classifier
	 *
	 * @return a description of the classifier suitable for displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Class for building and using a multinomial Naive Bayes classifier. " + "For more information see,\n\n" + this.getTechnicalInformation().toString() + "\n\n" + "The core equation for this classifier:\n\n"
				+ "P[Ci|D] = (P[D|Ci] x P[Ci]) / P[D] (Bayes' rule)\n\n" + "where Ci is class i and D is a document.";
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed information about the
	 * technical background of this class, e.g., paper reference or book this class is based on.
	 *
	 * @return the technical information about this class
	 */
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.INPROCEEDINGS);
		result.setValue(Field.AUTHOR, "Andrew Mccallum and Kamal Nigam");
		result.setValue(Field.YEAR, "1998");
		result.setValue(Field.TITLE, "A Comparison of Event Models for Naive Bayes Text Classification");
		result.setValue(Field.BOOKTITLE, "AAAI-98 Workshop on 'Learning for Text Categorization'");

		return result;
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
		result.enable(Capability.NUMERIC_ATTRIBUTES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		return result;
	}

	/**
	 * Sets up the classifier before any actual instances are processed.
	 */
	protected void initializeClassifier(final Instances instances) throws Exception {

		// can classifier handle the data?
		this.getCapabilities().testWithFail(instances);

		this.m_headerInfo = new Instances(instances, 0);
		this.m_numClasses = instances.numClasses();
		this.m_numAttributes = instances.numAttributes();
		this.m_probOfWordGivenClass = new double[this.m_numClasses][];

		// Initialize the matrix of word counts
		for (int c = 0; c < this.m_numClasses; c++) {
			this.m_probOfWordGivenClass[c] = new double[this.m_numAttributes];
			for (int att = 0; att < this.m_numAttributes; att++) {
				this.m_probOfWordGivenClass[c][att] = 1.0;
			}
		}

		// Initialize class counts
		this.m_probOfClass = new double[this.m_numClasses];
		for (int i = 0; i < this.m_numClasses; i++) {
			this.m_probOfClass[i] = 1.0;
		}
	}

	/**
	 * Generates the classifier.
	 *
	 * @param instances
	 *          set of instances serving as training data
	 * @throws Exception
	 *           if the classifier has not been generated successfully
	 */
	@Override
	public void buildClassifier(final Instances instances) throws Exception {

		this.initializeClassifier(instances);

		// enumerate through the instances
		double[] wordsPerClass = new double[this.m_numClasses];
		for (Instance instance : instances) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			double classValue = instance.value(instance.classIndex());
			if (!Utils.isMissingValue(classValue)) {
				int classIndex = (int) classValue;
				this.m_probOfClass[classIndex] += instance.weight();
				for (int a = 0; a < instance.numValues(); a++) {
					// XXX kill weka execution
					if (Thread.interrupted()) {
						throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
					}
					if (instance.index(a) != instance.classIndex()) {
						if (!instance.isMissingSparse(a)) {
							double numOccurrences = instance.valueSparse(a) * instance.weight();
							if (numOccurrences < 0) {
								throw new Exception("Numeric attribute values must all be greater or equal to zero.");
							}
							wordsPerClass[classIndex] += numOccurrences;
							this.m_probOfWordGivenClass[classIndex][instance.index(a)] += numOccurrences;
						}
					}
				}
			}
		}

		/*
		 * normalising probOfWordGivenClass values and saving each value as the log of each value
		 */
		for (int c = 0; c < this.m_numClasses; c++) {
			for (int v = 0; v < this.m_numAttributes; v++) {
				this.m_probOfWordGivenClass[c][v] = Math.log(this.m_probOfWordGivenClass[c][v]) - Math.log(wordsPerClass[c] + this.m_numAttributes - 1);
			}
		}

		// Normalize prior class probabilities
		Utils.normalize(this.m_probOfClass);
	}

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param instance
	 *          the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception
	 *           if there is a problem generating the prediction
	 */
	@Override
	public double[] distributionForInstance(final Instance instance) throws Exception {

		double[] probOfClassGivenDoc = new double[this.m_numClasses];

		// calculate the array of log(Pr[D|C])
		double[] logDocGivenClass = new double[this.m_numClasses];
		for (int h = 0; h < this.m_numClasses; h++) {
			// XXX kill weka execution
			if (Thread.interrupted()) {
				throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
			}
			logDocGivenClass[h] = this.probOfDocGivenClass(instance, h);
		}

		double max = logDocGivenClass[Utils.maxIndex(logDocGivenClass)];

		for (int i = 0; i < this.m_numClasses; i++) {
			probOfClassGivenDoc[i] = Math.exp(logDocGivenClass[i] - max) * this.m_probOfClass[i];
		}

		Utils.normalize(probOfClassGivenDoc);

		return probOfClassGivenDoc;
	}

	/**
	 * log(N!) + (sum for all the words i)(log(Pi^ni) - log(ni!))
	 *
	 * where N is the total number of words Pi is the probability of obtaining word i ni is the number
	 * of times the word at index i occurs in the document
	 *
	 * Actually, this method just computes (sum for all the words i)(log(Pi^ni) because the factorials
	 * are irrelevant when posterior class probabilities are computed.
	 *
	 * @param inst
	 *          The instance to be classified
	 * @param classIndex
	 *          The index of the class we are calculating the probability with respect to
	 *
	 * @return The log of the probability of the document occuring given the class
	 */

	protected double probOfDocGivenClass(final Instance inst, final int classIndex) {

		double answer = 0;

		for (int i = 0; i < inst.numValues(); i++) {
			if (inst.index(i) != inst.classIndex()) {
				answer += (inst.valueSparse(i) * this.m_probOfWordGivenClass[classIndex][inst.index(i)]);
			}
		}

		return answer;
	}

	/**
	 * Returns a string representation of the classifier.
	 *
	 * @return a string representation of the classifier
	 */
	@Override
	public String toString() {
		StringBuffer result = new StringBuffer("The independent probability of a class\n--------------------------------------\n");

		for (int c = 0; c < this.m_numClasses; c++) {
			result.append(this.m_headerInfo.classAttribute().value(c)).append("\t").append(Utils.doubleToString(this.m_probOfClass[c], this.getNumDecimalPlaces())).append("\n");
		}

		result.append("\nThe probability of a word given the class\n-----------------------------------------\n\t");

		for (int c = 0; c < this.m_numClasses; c++) {
			result.append(this.m_headerInfo.classAttribute().value(c)).append("\t");
		}

		result.append("\n");

		for (int w = 0; w < this.m_numAttributes; w++) {
			if (w != this.m_headerInfo.classIndex()) {
				result.append(this.m_headerInfo.attribute(w).name()).append("\t");
				for (int c = 0; c < this.m_numClasses; c++) {
					result.append(Utils.doubleToString(Math.exp(this.m_probOfWordGivenClass[c][w]), this.getNumDecimalPlaces())).append("\t");
				}
				result.append("\n");
			}
		}

		return result.toString();
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

	/**
	 * Main method for testing this class.
	 *
	 * @param argv
	 *          the options
	 */
	public static void main(final String[] argv) {
		runClassifier(new NaiveBayesMultinomial(), argv);
	}
}
