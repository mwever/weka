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
 * CheckSource.java
 * Copyright (C) 2007-2012 University of Waikato, Hamilton, New Zealand
 */

package weka.filters;

import java.io.File;
import java.util.Enumeration;
import java.util.Vector;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * A simple class for checking the source generated from Filters implementing the <code>weka.filters.Sourcable</code> interface. It takes a filter, the classname of the generated source and the dataset the source was generated with as
 * parameters and tests the output of the built filter against the output of the generated source. Use option '-h' to display all available commandline options.
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -W &lt;classname and options&gt;
 *  The filter (incl. options) that was used to generate
 *  the source code.
 * </pre>
 *
 * <pre>
 * -S &lt;classname&gt;
 *  The classname of the generated source code.
 * </pre>
 *
 * <pre>
 * -t &lt;file&gt;
 *  The training set with which the source code was generated.
 * </pre>
 *
 * <pre>
 * -c &lt;index&gt;
 *  The class index of the training set. 'first' and 'last' are
 *  valid indices.
 *  (default: none)
 * </pre>
 *
 * <!-- options-end -->
 *
 * Options after -- are passed to the designated filter.
 * <p>
 *
 * @author fracpete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 * @see weka.filters.Sourcable
 */
public class CheckSource implements OptionHandler, RevisionHandler {

	/** the classifier used for generating the source code */
	protected Filter m_Filter = null;

	/** the generated source code */
	protected Filter m_SourceCode = null;

	/** the dataset to use for testing */
	protected File m_Dataset = null;

	/** the class index */
	protected int m_ClassIndex = -1;

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>();

		result.addElement(new Option("\tThe filter (incl. options) that was used to generate\n" + "\tthe source code.", "W", 1, "-W <classname and options>"));

		result.addElement(new Option("\tThe classname of the generated source code.", "S", 1, "-S <classname>"));

		result.addElement(new Option("\tThe training set with which the source code was generated.", "t", 1, "-t <file>"));

		result.addElement(new Option("\tThe class index of the training set. 'first' and 'last' are\n" + "\tvalid indices.\n" + "\t(default: none)", "c", 1, "-c <index>"));

		return result.elements();
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 *
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 *
	 * <pre>
	 * -W &lt;classname and options&gt;
	 *  The filter (incl. options) that was used to generate
	 *  the source code.
	 * </pre>
	 *
	 * <pre>
	 * -S &lt;classname&gt;
	 *  The classname of the generated source code.
	 * </pre>
	 *
	 * <pre>
	 * -t &lt;file&gt;
	 *  The training set with which the source code was generated.
	 * </pre>
	 *
	 * <pre>
	 * -c &lt;index&gt;
	 *  The class index of the training set. 'first' and 'last' are
	 *  valid indices.
	 *  (default: none)
	 * </pre>
	 *
	 * <!-- options-end -->
	 *
	 * Options after -- are passed to the designated filter.
	 * <p>
	 *
	 * @param options
	 *            the list of options as an array of strings
	 * @throws Exception
	 *             if an option is not supported
	 */
	@Override
	public void setOptions(final String[] options) throws Exception {
		String tmpStr;
		String[] spec;
		String classname;

		tmpStr = Utils.getOption('W', options);
		if (tmpStr.length() > 0) {
			spec = Utils.splitOptions(tmpStr);
			if (spec.length == 0) {
				throw new IllegalArgumentException("Invalid filter specification string");
			}
			classname = spec[0];
			spec[0] = "";
			this.setFilter((Filter) Utils.forName(Filter.class, classname, spec));
		} else {
			throw new Exception("No filter (classname + options) provided!");
		}

		tmpStr = Utils.getOption('S', options);
		if (tmpStr.length() > 0) {
			spec = Utils.splitOptions(tmpStr);
			if (spec.length != 1) {
				throw new IllegalArgumentException("Invalid source code specification string");
			}
			classname = spec[0];
			spec[0] = "";
			this.setSourceCode((Filter) Utils.forName(Filter.class, classname, spec));
		} else {
			throw new Exception("No source code (classname) provided!");
		}

		tmpStr = Utils.getOption('t', options);
		if (tmpStr.length() != 0) {
			this.setDataset(new File(tmpStr));
		} else {
			throw new Exception("No dataset provided!");
		}

		tmpStr = Utils.getOption('c', options);
		if (tmpStr.length() != 0) {
			if (tmpStr.equals("first")) {
				this.setClassIndex(0);
			} else if (tmpStr.equals("last")) {
				this.setClassIndex(-2);
			} else {
				this.setClassIndex(Integer.parseInt(tmpStr) - 1);
			}
		} else {
			this.setClassIndex(-1);
		}
	}

	/**
	 * Gets the current settings of the filter.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String[] getOptions() {
		Vector<String> result;

		result = new Vector<String>();

		if (this.getFilter() != null) {
			result.add("-W");
			result.add(this.getFilter().getClass().getName() + " " + Utils.joinOptions(((OptionHandler) this.getFilter()).getOptions()));
		}

		if (this.getSourceCode() != null) {
			result.add("-S");
			result.add(this.getSourceCode().getClass().getName());
		}

		if (this.getDataset() != null) {
			result.add("-t");
			result.add(this.m_Dataset.getAbsolutePath());
		}

		if (this.getClassIndex() != -1) {
			result.add("-c");
			if (this.getClassIndex() == -2) {
				result.add("last");
			} else if (this.getClassIndex() == 0) {
				result.add("first");
			} else {
				result.add("" + (this.getClassIndex() + 1));
			}
		}

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Sets the filter to use for the comparison.
	 *
	 * @param value
	 *            the filter to use
	 */
	public void setFilter(final Filter value) {
		this.m_Filter = value;
	}

	/**
	 * Gets the filter being used for the tests, can be null.
	 *
	 * @return the currently set filter
	 */
	public Filter getFilter() {
		return this.m_Filter;
	}

	/**
	 * Sets the class to test.
	 *
	 * @param value
	 *            the class to test
	 */
	public void setSourceCode(final Filter value) {
		this.m_SourceCode = value;
	}

	/**
	 * Gets the class to test.
	 *
	 * @return the currently set class, can be null.
	 */
	public Filter getSourceCode() {
		return this.m_SourceCode;
	}

	/**
	 * Sets the dataset to use for testing.
	 *
	 * @param value
	 *            the dataset to use.
	 */
	public void setDataset(final File value) {
		if (!value.exists()) {
			throw new IllegalArgumentException("Dataset '" + value.getAbsolutePath() + "' does not exist!");
		} else {
			this.m_Dataset = value;
		}
	}

	/**
	 * Gets the dataset to use for testing, can be null.
	 *
	 * @return the dataset to use.
	 */
	public File getDataset() {
		return this.m_Dataset;
	}

	/**
	 * Sets the class index of the dataset.
	 *
	 * @param value
	 *            the class index of the dataset.
	 */
	public void setClassIndex(final int value) {
		this.m_ClassIndex = value;
	}

	/**
	 * Gets the class index of the dataset.
	 *
	 * @return the current class index.
	 */
	public int getClassIndex() {
		return this.m_ClassIndex;
	}

	/**
	 * compares two Instance
	 *
	 * @param inst1
	 *            the first Instance object to compare
	 * @param inst2
	 *            the second Instance object to compare
	 * @return true if both are the same
	 */
	protected boolean compare(final Instance inst1, final Instance inst2) {
		boolean result;
		int i;

		// check dimension
		result = (inst1.numAttributes() == inst2.numAttributes());

		// check content
		if (result) {
			for (i = 0; i < inst1.numAttributes(); i++) {
				if (Double.isNaN(inst1.value(i)) && (Double.isNaN(inst2.value(i)))) {
					continue;
				}

				if (inst1.value(i) != inst2.value(i)) {
					result = false;
					break;
				}
			}
		}

		return result;
	}

	/**
	 * compares the two Instances objects
	 *
	 * @param inst1
	 *            the first Instances object to compare
	 * @param inst2
	 *            the second Instances object to compare
	 * @return true if both are the same
	 */
	protected boolean compare(final Instances inst1, final Instances inst2) {
		boolean result;
		int i;

		// check dimensions
		result = (inst1.numInstances() == inst2.numInstances());

		// check content
		if (result) {
			for (i = 0; i < inst1.numInstances(); i++) {
				result = this.compare(inst1.instance(i), inst2.instance(i));
				if (!result) {
					break;
				}
			}
		}

		return result;
	}

	/**
	 * performs the comparison test
	 *
	 * @return true if tests were successful
	 * @throws Exception
	 *             if tests fail
	 */
	public boolean execute() throws Exception {
		boolean result;
		Instances data;
		Instance filteredInstance;
		Instances filteredInstances;
		Instance filteredInstanceSource;
		Instances filteredInstancesSource;
		DataSource source;
		Filter filter;
		Filter filterSource;
		int i;

		result = true;

		// a few checks
		if (this.getFilter() == null) {
			throw new Exception("No filter set!");
		}
		if (this.getSourceCode() == null) {
			throw new Exception("No source code set!");
		}
		if (this.getDataset() == null) {
			throw new Exception("No dataset set!");
		}
		if (!this.getDataset().exists()) {
			throw new Exception("Dataset '" + this.getDataset().getAbsolutePath() + "' does not exist!");
		}

		// load data
		source = new DataSource(this.getDataset().getAbsolutePath());
		data = source.getDataSet();
		if (this.getClassIndex() == -2) {
			data.setClassIndex(data.numAttributes() - 1);
		} else {
			data.setClassIndex(this.getClassIndex());
		}

		// compare output
		// 1. batch filtering
		filter = Filter.makeCopy(this.getFilter());
		filter.setInputFormat(data);
		filteredInstances = Filter.useFilter(data, filter);

		filterSource = Filter.makeCopy(this.getSourceCode());
		filterSource.setInputFormat(data);
		filteredInstancesSource = Filter.useFilter(data, filterSource);

		result = this.compare(filteredInstances, filteredInstancesSource);

		// 2. instance by instance
		if (result) {
			filter = Filter.makeCopy(this.getFilter());
			filter.setInputFormat(data);
			Filter.useFilter(data, filter);

			filterSource = Filter.makeCopy(this.getSourceCode());
			filterSource.setInputFormat(data);

			for (i = 0; i < data.numInstances(); i++) {
				filter.input(data.instance(i));
				filter.batchFinished();
				filteredInstance = filter.output();

				filterSource.input(data.instance(i));
				filterSource.batchFinished();
				filteredInstanceSource = filterSource.output();

			}
		}

		return result;
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
	 * Executes the tests, use "-h" to list the commandline options.
	 *
	 * @param args
	 *            the commandline parameters
	 * @throws Exception
	 *             if something goes wrong
	 */
	public static void main(final String[] args) throws Exception {
		CheckSource check;
		StringBuffer text;
		Enumeration<Option> enm;

		check = new CheckSource();
		if (Utils.getFlag('h', args)) {
			text = new StringBuffer();
			text.append("\nHelp requested:\n\n");
			enm = check.listOptions();
			while (enm.hasMoreElements()) {
				Option option = enm.nextElement();
				text.append(option.synopsis() + "\n");
				text.append(option.description() + "\n");
			}
		} else {
			check.setOptions(args);
			if (check.execute()) {
				System.out.println("Tests OK!");
			} else {
				System.out.println("Tests failed!");
			}
		}
	}
}
