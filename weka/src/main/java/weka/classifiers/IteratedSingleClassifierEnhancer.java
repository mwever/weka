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
 *    IteratedSingleClassifierEnhancer.java
 *    Copyright (C) 2004-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

/**
 * Abstract utility class for handling settings common to
 * meta classifiers that build an ensemble from a single base learner.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */
public abstract class IteratedSingleClassifierEnhancer extends SingleClassifierEnhancer {

	/** for serialization */
	private static final long serialVersionUID = -6217979135443319724L;

	/** Array for storing the generated base classifiers. */
	protected Classifier[] m_Classifiers;

	/** The number of iterations. */
	protected int m_NumIterations = this.defaultNumberOfIterations();

	/**
	 * The default number of iterations to perform.
	 */
	protected int defaultNumberOfIterations() {
		return 10;
	}

	public Classifier[] getM_Classifiers() {
		return this.m_Classifiers;
	}

	/**
	 * Stump method for building the classifiers.
	 *
	 * @param data the training data to be used for generating the
	 * bagged classifier.
	 * @exception Exception if the classifier could not be built successfully
	 */
	@Override
	public void buildClassifier(final Instances data) throws Exception {

		if (this.m_Classifier == null) {
			throw new Exception("A base classifier has not been specified!");
		}
		this.m_Classifiers = AbstractClassifier.makeCopies(this.m_Classifier, this.m_NumIterations);
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>(2);

		newVector.addElement(new Option("\tNumber of iterations.\n" + "\t(current value " + this.getNumIterations() + ")", "I", 1, "-I <num>"));

		newVector.addAll(Collections.list(super.listOptions()));

		return newVector.elements();
	}

	/**
	 * Parses a given list of options. Valid options are:<p>
	 *
	 * -W classname <br>
	 * Specify the full class name of the base learner.<p>
	 *
	 * -I num <br>
	 * Set the number of iterations (default 10). <p>
	 *
	 * Options after -- are passed to the designated classifier.<p>
	 *
	 * @param options the list of options as an array of strings
	 * @exception Exception if an option is not supported
	 */
	@Override
	public void setOptions(final String[] options) throws Exception {

		String iterations = Utils.getOption('I', options);
		if (iterations.length() != 0) {
			this.setNumIterations(Integer.parseInt(iterations));
		} else {
			this.setNumIterations(this.defaultNumberOfIterations());
		}

		super.setOptions(options);
	}

	/**
	 * Gets the current settings of the classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String[] getOptions() {

		String[] superOptions = super.getOptions();
		String[] options = new String[superOptions.length + 2];

		int current = 0;
		options[current++] = "-I";
		options[current++] = "" + this.getNumIterations();

		System.arraycopy(superOptions, 0, options, current, superOptions.length);

		return options;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String numIterationsTipText() {
		return "The number of iterations to be performed.";
	}

	/**
	 * Sets the number of bagging iterations
	 */
	public void setNumIterations(final int numIterations) {

		this.m_NumIterations = numIterations;
	}

	/**
	 * Gets the number of bagging iterations
	 *
	 * @return the maximum number of bagging iterations
	 */
	public int getNumIterations() {

		return this.m_NumIterations;
	}
}
