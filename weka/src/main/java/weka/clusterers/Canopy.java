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
 *    Canopy.java
 *    Copyright (C) 2014 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.clusterers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.rules.DecisionTableHashKey;
import weka.core.AttributeStats;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NormalizableDistance;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SparseInstance;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 <!-- globalinfo-start -->
 * Cluster data using the capopy clustering algorithm, which requires just one pass over the data. Can run in eitherbatch or incremental mode. Results are generally not as good when running incrementally as the min/max for each numeric attribute is not known in advance. Has a heuristic (based on attribute std. deviations), that can be used in batch mode, for setting the T2 distance. The T2 distance determines how many canopies (clusters) are formed. When the user specifies a specific number (N) of clusters to generate, the algorithm will return the top N canopies (as determined by T2 density) when N &lt; number of canopies (this applies to both batch and incremental learning); when N &gt; number of canopies, the difference is made up by selecting training instances randomly (this can only be done when batch training). For more information see:<br/>
 * <br/>
 * A. McCallum, K. Nigam, L.H. Ungar: Efficient Clustering of High Dimensional Data Sets with Application to Reference Matching. In: Proceedings of the sixth ACM SIGKDD internation conference on knowledge discovery and data mining ACM-SIAM symposium on Discrete algorithms, 169-178, 2000.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{McCallum2000,
 *    author = {A. McCallum and K. Nigam and L.H. Ungar},
 *    booktitle = {Proceedings of the sixth ACM SIGKDD internation conference on knowledge discovery and data mining ACM-SIAM symposium on Discrete algorithms},
 *    pages = {169-178},
 *    title = {Efficient Clustering of High Dimensional Data Sets with Application to Reference Matching},
 *    year = {2000}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 *
 * <pre> -N &lt;num&gt;
 *  Number of clusters.
 *  (default 2).</pre>
 *
 * <pre> -max-candidates &lt;num&gt;
 *  Maximum number of candidate canopies to retain in memory
 *  at any one time. T2 distance plus, data characteristics,
 *  will determine how many candidate canopies are formed before
 *  periodic and final pruning are performed, which might result
 *  in exceess memory consumption. This setting avoids large numbers
 *  of candidate canopies consuming memory. (default = 100)</pre>
 *
 * <pre> -periodic-pruning &lt;num&gt;
 *  How often to prune low density canopies.
 *  (default = every 10,000 training instances)</pre>
 *
 * <pre> -min-density
 *  Minimum canopy density, below which a canopy will be pruned
 *  during periodic pruning. (default = 2 instances)</pre>
 *
 * <pre> -t2
 *  The T2 distance to use. Values &lt; 0 indicate that
 *  a heuristic based on attribute std. deviation should be used to set this.
 *  Note that this heuristic can only be used when batch training
 *  (default = -1.0)</pre>
 *
 * <pre> -t1
 *  The T1 distance to use. A value &lt; 0 is taken as a
 *  positive multiplier for T2. (default = -1.5)</pre>
 *
 * <pre> -M
 *  Don't replace missing values with mean/mode when running in batch mode.
 * </pre>
 *
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 *
 * <pre> -output-debug-info
 *  If set, clusterer is run in debug mode and
 *  may output additional info to the console</pre>
 *
 * <pre> -do-not-check-capabilities
 *  If set, clusterer capabilities are not checked before clusterer is built
 *  (use with caution).</pre>
 *
 <!-- options-end -->
 *
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @version $Revision$
 */
public class Canopy extends RandomizableClusterer implements UpdateableClusterer, NumberOfClustersRequestable, OptionHandler, TechnicalInformationHandler {

	/** For serialization */
	private static final long serialVersionUID = 2067574593448223334L;

	/** The canopy centers */
	protected Instances m_canopies;

	/** The T2 density of each canopy */
	protected List<double[]> m_canopyT2Density;
	protected List<double[][]> m_canopyCenters;
	protected List<double[]> m_canopyNumMissingForNumerics;

	/**
	 * The list of canopies that each canopy is a member of (according to the T1
	 * radius, which can overlap). Each bit position in the long values
	 * corresponds to one canopy. Outer list order corresponds to the order of the
	 * instances that store the actual canopy centers
	 */
	protected List<long[]> m_clusterCanopies;

	public static final double DEFAULT_T2 = -1.0;
	public static final double DEFAULT_T1 = -1.25;

	/** < 0 means use the heuristic based on std. dev. to set the t2 radius */
	protected double m_userT2 = DEFAULT_T2;

	/**
	 * < 0 indicates the multiplier to use for T2 when setting T1, otherwise the
	 * value is take as is
	 */
	protected double m_userT1 = DEFAULT_T1;

	/** Outer radius */
	protected double m_t1 = this.m_userT1;

	/** Inner radius */
	protected double m_t2 = this.m_userT2;

	/**
	 * Prune low-density candidate canopies after every x instances have been seen
	 */
	protected int m_periodicPruningRate = 10000;

	/**
	 * The minimum cluster density (according to T2 distance) allowed. Used when
	 * periodically pruning candidate canopies
	 */
	protected double m_minClusterDensity = 2;

	/** The maximum number of candidate canopies to hold in memory at any one time */
	protected int m_maxCanopyCandidates = 100;

	/**
	 * True if the pruning operation did remove at least one low density canopy
	 * the last time it was invoked
	 */
	protected boolean m_didPruneLastTime = true;

	/** Number of training instances seen so far */
	protected int m_instanceCount;

	/**
	 * Default is to let the t2 radius determine how many canopies/clusters are
	 * formed
	 */
	protected int m_numClustersRequested = -1;

	/**
	 * If not null, then this is expected to be a filter that can replace missing
	 * values immediately (at training and testing time)
	 */
	protected Filter m_missingValuesReplacer;

	/**
	 * Replace missing values globally when running in batch mode?
	 */
	protected boolean m_dontReplaceMissing = false;

	/** The distance function to use */
	protected NormalizableDistance m_distanceFunction = new EuclideanDistance();

	/**
	 * Used to pad out number of cluster centers if fewer canopies are generated
	 * than the number of requested clusters and we are running in batch mode.
	 */
	protected Instances m_trainingData;

	/**
	 * Returns a string describing this clusterer.
	 *
	 * @return a description of the evaluator suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Cluster data using the capopy clustering algorithm, which requires just " + "one pass over the data. Can run in either" + "batch or incremental mode. Results are generally not as good when "
				+ "running incrementally as the min/max for each numeric attribute is not " + "known in advance. Has a heuristic (based on attribute std. deviations), "
				+ "that can be used in batch mode, for setting the T2 distance. The T2 distance " + "determines how many canopies (clusters) are formed. When the user specifies "
				+ "a specific number (N) of clusters to generate, the algorithm will return the " + "top N canopies (as determined by T2 density) when N < number of canopies " + "(this applies to both batch and incremental learning); "
				+ "when N > number of canopies, the difference is made up by selecting training " + "instances randomly (this can only be done when batch training). For more " + "information see:\n\n"
				+ this.getTechnicalInformation().toString();

	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.INPROCEEDINGS);
		result.setValue(Field.AUTHOR, "A. McCallum and K. Nigam and L.H. Ungar");
		result.setValue(Field.TITLE, "Efficient Clustering of High Dimensional Data Sets with Application to Reference Matching");
		result.setValue(Field.BOOKTITLE, "Proceedings of the sixth ACM SIGKDD internation conference on " + "knowledge discovery and data mining " + "ACM-SIAM symposium on Discrete algorithms");
		result.setValue(Field.YEAR, "2000");
		result.setValue(Field.PAGES, "169-178");

		return result;
	}

	/**
	 * Returns default capabilities of the clusterer.
	 *
	 * @return the capabilities of this clusterer
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();
		result.enable(Capability.NO_CLASS);

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		return result;
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>();

		result.addElement(new Option("\tNumber of clusters.\n" + "\t(default 2).", "N", 1, "-N <num>"));

		result.addElement(new Option(
				"\tMaximum number of candidate canopies to retain in memory\n\t" + "at any one time. T2 distance plus, data characteristics,\n\t" + "will determine how many candidate canopies are formed before\n\t"
						+ "periodic and final pruning are performed, which might result\n\t" + "in exceess memory consumption. This setting avoids large numbers\n\t" + "of candidate canopies consuming memory. (default = 100)",
				"-max-candidates", 1, "-max-candidates <num>"));

		result.addElement(new Option("\tHow often to prune low density canopies. \n\t" + "(default = every 10,000 training instances)", "periodic-pruning", 1, "-periodic-pruning <num>"));

		result.addElement(new Option("\tMinimum canopy density, below which a canopy will be pruned\n\t" + "during periodic pruning. (default = 2 instances)", "min-density", 1, "-min-density"));

		result.addElement(new Option("\tThe T2 distance to use. Values < 0 indicate that\n\t" + "a heuristic based on attribute std. deviation should be used to set this.\n\t"
				+ "Note that this heuristic can only be used when batch training\n\t" + "(default = -1.0)", "t2", 1, "-t2"));

		result.addElement(new Option("\tThe T1 distance to use. A value < 0 is taken as a\n\t" + "positive multiplier for T2. (default = -1.5)", "t1", 1, "-t1"));

		result.addElement(new Option("\tDon't replace missing values with mean/mode when " + "running in batch mode.\n", "M", 0, "-M"));

		result.addAll(Collections.list(super.listOptions()));

		return result.elements();
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 *
	 <!-- options-start -->
	 * Valid options are: <p/>
	 *
	 * <pre> -N &lt;num&gt;
	 *  Number of clusters.
	 *  (default 2).</pre>
	 *
	 * <pre> -max-candidates &lt;num&gt;
	 *  Maximum number of candidate canopies to retain in memory
	 *  at any one time. T2 distance plus, data characteristics,
	 *  will determine how many candidate canopies are formed before
	 *  periodic and final pruning are performed, which might result
	 *  in exceess memory consumption. This setting avoids large numbers
	 *  of candidate canopies consuming memory. (default = 100)</pre>
	 *
	 * <pre> -periodic-pruning &lt;num&gt;
	 *  How often to prune low density canopies.
	 *  (default = every 10,000 training instances)</pre>
	 *
	 * <pre> -min-density
	 *  Minimum canopy density, below which a canopy will be pruned
	 *  during periodic pruning. (default = 2 instances)</pre>
	 *
	 * <pre> -t2
	 *  The T2 distance to use. Values &lt; 0 indicate that
	 *  a heuristic based on attribute std. deviation should be used to set this.
	 *  Note that this heuristic can only be used when batch training
	 *  (default = -1.0)</pre>
	 *
	 * <pre> -t1
	 *  The T1 distance to use. A value &lt; 0 is taken as a
	 *  positive multiplier for T2. (default = -1.5)</pre>
	 *
	 * <pre> -M
	 *  Don't replace missing values with mean/mode when running in batch mode.
	 * </pre>
	 *
	 * <pre> -S &lt;num&gt;
	 *  Random number seed.
	 *  (default 1)</pre>
	 *
	 * <pre> -output-debug-info
	 *  If set, clusterer is run in debug mode and
	 *  may output additional info to the console</pre>
	 *
	 * <pre> -do-not-check-capabilities
	 *  If set, clusterer capabilities are not checked before clusterer is built
	 *  (use with caution).</pre>
	 *
	 <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings throws Exception
	 *          if an option is not supported
	 */
	@Override
	public void setOptions(final String[] options) throws Exception {

		String temp = Utils.getOption('N', options);
		if (temp.length() > 0) {
			this.setNumClusters(Integer.parseInt(temp));
		}

		temp = Utils.getOption("max-candidates", options);
		if (temp.length() > 0) {
			this.setMaxNumCandidateCanopiesToHoldInMemory(Integer.parseInt(temp));
		}

		temp = Utils.getOption("periodic-pruning", options);
		if (temp.length() > 0) {
			this.setPeriodicPruningRate(Integer.parseInt(temp));
		}

		temp = Utils.getOption("min-density", options);
		if (temp.length() > 0) {
			this.setMinimumCanopyDensity(Double.parseDouble(temp));
		}

		temp = Utils.getOption("t2", options);
		if (temp.length() > 0) {
			this.setT2(Double.parseDouble(temp));
		}

		temp = Utils.getOption("t1", options);
		if (temp.length() > 0) {
			this.setT1(Double.parseDouble(temp));
		}

		this.setDontReplaceMissingValues(Utils.getFlag('M', options));

		super.setOptions(options);
	}

	/**
	 * Gets the current settings of Canopy.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	@Override
	public String[] getOptions() {

		Vector<String> result = new Vector<String>();

		result.add("-N");
		result.add("" + this.getNumClusters());
		result.add("-max-candidates");
		result.add("" + this.getMaxNumCandidateCanopiesToHoldInMemory());
		result.add("-periodic-pruning");
		result.add("" + this.getPeriodicPruningRate());
		result.add("-min-density");
		result.add("" + this.getMinimumCanopyDensity());
		result.add("-t2");
		result.add("" + this.getT2());
		result.add("-t1");
		result.add("" + this.getT1());

		if (this.getDontReplaceMissingValues()) {
			result.add("-M");
		}

		Collections.addAll(result, super.getOptions());

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Tests if two sets of canopies have a non-empty intersection
	 *
	 * @param first the first canopy set
	 * @param second the second canopy set
	 * @return true if the intersection is non-empty
	 * @throws Exception if a problem occurs
	 */
	public static boolean nonEmptyCanopySetIntersection(final long[] first, final long[] second) throws Exception {
		if (first.length != second.length) {
			throw new Exception("Canopy lists need to be the same length");
		}

		if (first.length == 0 || second.length == 0) {
			return false;
		}

		for (int i = 0; i < first.length; i++) {
			long firstBlock = first[i];
			long secondBlock = second[i];

			if ((firstBlock & secondBlock) != 0L) {
				return true;
			}
		}

		return false;
	}

	private static void updateCanopyAssignment(final long[] assigned, final int toAssign) {
		int whichLong = toAssign / 64;
		int whichBitPosition = toAssign % 64;
		long mask = 1L << whichBitPosition;

		assigned[whichLong] |= mask;
	}

	/**
	 * Uses T1 distance to assign canopies to the supplied instance. If the
	 * instance does not fall within T1 distance of any canopies then the instance
	 * has the closest canopy assigned to it.
	 *
	 * @param inst the instance to find covering canopies for
	 * @return a set of canopies that contain this instance according to T1
	 *         distance
	 * @throws Exception if a problem occurs
	 */
	public long[] assignCanopies(Instance inst) throws Exception {
		if (this.m_missingValuesReplacer != null) {
			this.m_missingValuesReplacer.input(inst);
			inst = this.m_missingValuesReplacer.output();
		}

		int numLongs = this.m_canopies.size() / 64 + 1;
		long[] assigned = new long[numLongs];

		double minDist = Double.MAX_VALUE;
		double bitsSet = 0;
		int index = -1;
		for (int i = 0; i < this.m_canopies.numInstances(); i++) {
			double dist = this.m_distanceFunction.distance(inst, this.m_canopies.instance(i));
			if (dist < minDist) {
				minDist = dist;
				index = i;
			}

			if (dist < this.m_t1) {
				updateCanopyAssignment(assigned, i);
				bitsSet++;
				// assigned.add(i);
			}
		}

		// this won't be necessary (for the training data) unless canopies have been
		// pruned due to the user requesting fewer canopies than are created
		// naturally according to the t2 radius
		if (bitsSet == 0) {
			// add the closest canopy
			updateCanopyAssignment(assigned, index);
		}

		return assigned;
	}

	protected void updateCanopyCenter(final Instance newInstance, final double[][] center, final double[] numMissingNumerics) {
		for (int i = 0; i < newInstance.numAttributes(); i++) {
			if (newInstance.attribute(i).isNumeric()) {
				if (center[i].length == 0) {
					center[i] = new double[1];
				}

				if (!newInstance.isMissing(i)) {
					center[i][0] += newInstance.value(i);
				} else {
					numMissingNumerics[i]++;
				}
			} else if (newInstance.attribute(i).isNominal()) {
				if (center[i].length == 0) {
					// +1 for missing
					center[i] = new double[newInstance.attribute(i).numValues() + 1];
				}
				if (newInstance.isMissing(i)) {
					center[i][center[i].length - 1]++;
				} else {
					center[i][(int) newInstance.value(i)]++;
				}
			}
		}
	}

	@Override
	public void updateClusterer(Instance newInstance) throws Exception {

		if (this.m_instanceCount > 0) {
			if (this.m_instanceCount % this.m_periodicPruningRate == 0) {
				this.pruneCandidateCanopies();
			}
		}

		this.m_instanceCount++;
		if (this.m_missingValuesReplacer != null) {
			this.m_missingValuesReplacer.input(newInstance);
			newInstance = this.m_missingValuesReplacer.output();
		}

		this.m_distanceFunction.update(newInstance);
		boolean addPoint = true;

		for (int i = 0; i < this.m_canopies.numInstances(); i++) {
			if (this.m_distanceFunction.distance(newInstance, this.m_canopies.instance(i)) < this.m_t2) {
				double[] density = this.m_canopyT2Density.get(i);
				density[0]++;
				addPoint = false;

				double[][] center = this.m_canopyCenters.get(i);
				double[] numMissingNumerics = this.m_canopyNumMissingForNumerics.get(i);
				this.updateCanopyCenter(newInstance, center, numMissingNumerics);

				break;
			}
		}

		if (addPoint && this.m_canopies.numInstances() < this.m_maxCanopyCandidates) {
			this.m_canopies.add(newInstance);
			double[] density = new double[1];
			density[0] = 1.0;
			this.m_canopyT2Density.add(density);

			double[][] center = new double[newInstance.numAttributes()][0];
			double[] numMissingNumerics = new double[newInstance.numAttributes()];
			this.updateCanopyCenter(newInstance, center, numMissingNumerics);
			this.m_canopyCenters.add(center);
			this.m_canopyNumMissingForNumerics.add(numMissingNumerics);
		}
	}

	/**
	 * Prune low density candidate canopies
	 */
	protected void pruneCandidateCanopies() {
		if (this.m_didPruneLastTime == false && this.m_canopies.size() == this.m_maxCanopyCandidates) {
			return;
		}

		this.m_didPruneLastTime = false;
		for (int i = this.m_canopies.numInstances() - 1; i >= 0; i--) {
			double dens = this.m_canopyT2Density.get(i)[0];
			if (dens < this.m_minClusterDensity) {
				double[] tempDens = this.m_canopyT2Density.remove(this.m_canopyT2Density.size() - 1);
				if (i < this.m_canopyT2Density.size()) {
					this.m_canopyT2Density.set(i, tempDens);
				}
				if (this.getDebug()) {
					System.err.println("Pruning a candidate canopy with density: " + dens);
				}
				this.m_didPruneLastTime = true;

				double[][] tempCenter = this.m_canopyCenters.remove(this.m_canopyCenters.size() - 1);
				if (i < this.m_canopyCenters.size()) {
					this.m_canopyCenters.set(i, tempCenter);
				}

				double[] tempNumMissingNumerics = this.m_canopyNumMissingForNumerics.remove(this.m_canopyNumMissingForNumerics.size() - 1);

				if (i < this.m_canopyNumMissingForNumerics.size()) {
					this.m_canopyNumMissingForNumerics.set(i, tempNumMissingNumerics);
				}

				if (i != this.m_canopies.numInstances() - 1) {
					this.m_canopies.swap(i, this.m_canopies.numInstances() - 1);
				}
				this.m_canopies.delete(this.m_canopies.numInstances() - 1);
			}
		}
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		if (this.m_canopies == null || this.m_canopies.size() == 0) {
			throw new Exception("No canopies available to cluster with!");
		}

		double[] d = new double[this.numberOfClusters()];

		if (this.m_missingValuesReplacer != null) {
			this.m_missingValuesReplacer.input(instance);
			instance = this.m_missingValuesReplacer.output();
		}

		for (int i = 0; i < this.m_canopies.numInstances(); i++) {
			double distance = this.m_distanceFunction.distance(instance, this.m_canopies.instance(i));

			d[i] = 1.0 / (1.0 + distance);
		}

		Utils.normalize(d);
		return d;
	}

	private void assignCanopiesToCanopyCenters() {
		// assign canopies to each canopy center
		this.m_clusterCanopies = new ArrayList<long[]>();
		for (int i = 0; i < this.m_canopies.size(); i++) {
			Instance inst = this.m_canopies.instance(i);
			try {
				long[] assignments = this.assignCanopies(inst);
				this.m_clusterCanopies.add(assignments);
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}
	}

	/**
	 * Adjust the final number of canopies to match the user-requested number (if
	 * possible)
	 *
	 * @param densities the density of each of the canopies
	* @throws InterruptedException
	 */
	protected void adjustCanopies(final double[] densities) throws InterruptedException {
		if (this.m_numClustersRequested < 0) {
			this.assignCanopiesToCanopyCenters();

			this.m_trainingData = new Instances(this.m_canopies, 0);
			return;
		}

		// more canopies than requested?
		if (this.m_canopies.numInstances() > this.m_numClustersRequested) {
			int[] sortedIndexes = Utils.stableSort(densities);

			Instances finalCanopies = new Instances(this.m_canopies, 0);
			int count = 0;
			for (int i = sortedIndexes.length - 1; count < this.m_numClustersRequested; i--) {
				finalCanopies.add(this.m_canopies.instance(sortedIndexes[i]));
				count++;
			}

			this.m_canopies = finalCanopies;
			List<double[][]> tempCanopyCenters = new ArrayList<double[][]>();
			List<double[]> tempT2Dists = new ArrayList<double[]>();
			List<double[]> tempMissings = new ArrayList<double[]>();

			// make sure that the center sums, densities and missing counts are
			// aligned with the new canopy list
			count = 0;
			for (int i = sortedIndexes.length - 1; count < finalCanopies.numInstances(); i--) {
				tempCanopyCenters.add(this.m_canopyCenters.get(sortedIndexes[i]));
				tempT2Dists.add(this.m_canopyT2Density.get(sortedIndexes[i]));
				tempMissings.add(this.m_canopyNumMissingForNumerics.get(sortedIndexes[i]));
				count++;
			}
			this.m_canopyCenters = tempCanopyCenters;
			this.m_canopyT2Density = tempT2Dists;
			this.m_canopyNumMissingForNumerics = tempMissings;

		} else if (this.m_canopies.numInstances() < this.m_numClustersRequested && this.m_trainingData != null && this.m_trainingData.numInstances() > 0) {

			// make up the difference with randomly selected instances (if possible)
			Random r = new Random(this.getSeed());
			for (int i = 0; i < 10; i++) {
				r.nextInt();
			}
			HashMap<DecisionTableHashKey, Integer> initC = new HashMap<DecisionTableHashKey, Integer>();
			DecisionTableHashKey hk = null;

			// put the existing canopies in the lookup
			for (int i = 0; i < this.m_canopies.numInstances(); i++) {
				try {
					hk = new DecisionTableHashKey(this.m_canopies.instance(i), this.m_canopies.numAttributes(), true);

					initC.put(hk, null);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}

			for (int j = this.m_trainingData.numInstances() - 1; j >= 0; j--) {
				int instIndex = r.nextInt(j + 1);
				try {
					hk = new DecisionTableHashKey(this.m_trainingData.instance(instIndex), this.m_trainingData.numAttributes(), true);
				} catch (Exception e) {
					e.printStackTrace();
				}
				if (!initC.containsKey(hk)) {
					Instance newInstance = this.m_trainingData.instance(instIndex);
					this.m_canopies.add(newInstance);

					double[] density = new double[1];
					density[0] = 1.0;
					this.m_canopyT2Density.add(density);

					double[][] center = new double[newInstance.numAttributes()][0];
					double[] numMissingNumerics = new double[newInstance.numAttributes()];
					this.updateCanopyCenter(newInstance, center, numMissingNumerics);
					this.m_canopyCenters.add(center);
					this.m_canopyNumMissingForNumerics.add(numMissingNumerics);

					initC.put(hk, null);
				}
				this.m_trainingData.swap(j, instIndex);

				if (this.m_canopies.numInstances() == this.m_numClustersRequested) {
					break;
				}
			}
		}

		this.assignCanopiesToCanopyCenters();

		// save memory
		this.m_trainingData = new Instances(this.m_canopies, 0);
	}

	@Override
	public void updateFinished() throws InterruptedException {
		if (this.m_canopies == null || this.m_canopies.numInstances() == 0) {
			return;
		}

		this.pruneCandidateCanopies();

		// set the final canopy centers and weights
		double[] densities = new double[this.m_canopies.size()];
		for (int i = 0; i < this.m_canopies.numInstances(); i++) {
			double[] density = this.m_canopyT2Density.get(i);
			double[][] centerSums = this.m_canopyCenters.get(i);
			double[] numMissingForNumerics = this.m_canopyNumMissingForNumerics.get(i);
			double[] finalCenter = new double[this.m_canopies.numAttributes()];
			for (int j = 0; j < this.m_canopies.numAttributes(); j++) {
				if (this.m_canopies.attribute(j).isNumeric()) {
					if (numMissingForNumerics[j] == density[0]) {
						finalCenter[j] = Utils.missingValue();
					} else {
						finalCenter[j] = centerSums[j][0] / (density[0] - numMissingForNumerics[j]);
					}
				} else if (this.m_canopies.attribute(j).isNominal()) {
					int mode = Utils.maxIndex(centerSums[j]);
					if (mode == centerSums[j].length - 1) {
						finalCenter[j] = Utils.missingValue();
					} else {
						finalCenter[j] = mode;
					}
				}
			}

			Instance finalCenterInst = this.m_canopies.instance(i) instanceof SparseInstance ? new SparseInstance(1.0, finalCenter) : new DenseInstance(1.0, finalCenter);
			this.m_canopies.set(i, finalCenterInst);

			this.m_canopies.instance(i).setWeight(density[0]);
			densities[i] = density[0];
		}

		this.adjustCanopies(densities);
	}

	/**
	 * Initialize the distance function (i.e set min/max values for numeric
	 * attributes) with the supplied instances.
	 *
	 * @param init the instances to initialize with
	 * @throws Exception if a problem occurs
	 */
	public void initializeDistanceFunction(Instances init) throws Exception {
		if (this.m_missingValuesReplacer != null) {
			init = Filter.useFilter(init, this.m_missingValuesReplacer);
		}

		this.m_distanceFunction.setInstances(init);
	}

	/**
	 * Pretty hokey heuristic to try and set t2 distance automatically based on
	 * standard deviation
	 *
	 * @param trainingBatch the training instances
	 * @throws Exception if a problem occurs
	 */
	protected void setT2T1BasedOnStdDev(final Instances trainingBatch) throws Exception {
		double normalizedStdDevSum = 0;

		for (int i = 0; i < trainingBatch.numAttributes(); i++) {
			if (trainingBatch.attribute(i).isNominal()) {
				normalizedStdDevSum += 0.25;
			} else if (trainingBatch.attribute(i).isNumeric()) {
				AttributeStats stats = trainingBatch.attributeStats(i);
				if (trainingBatch.numInstances() - stats.missingCount > 2) {
					double stdDev = stats.numericStats.stdDev;
					double min = stats.numericStats.min;
					double max = stats.numericStats.max;
					if (!Utils.isMissingValue(stdDev) && max - min > 0) {
						stdDev = 0.5 * stdDev / (max - min);
						normalizedStdDevSum += stdDev;
					}
				}
			}
		}

		normalizedStdDevSum = Math.sqrt(normalizedStdDevSum);
		if (normalizedStdDevSum > 0) {
			this.m_t2 = normalizedStdDevSum;
		}
	}

	@Override
	public void buildClusterer(Instances data) throws Exception {
		this.m_t1 = this.m_userT1;
		this.m_t2 = this.m_userT2;

		if (data.numInstances() == 0 && this.m_userT2 < 0) {
			System.err.println("The heuristic for setting T2 based on std. dev. can't be used when " + "running in incremental mode. Using default of 1.0.");
			this.m_t2 = 1.0;
		}

		this.m_canopyT2Density = new ArrayList<double[]>();
		this.m_canopyCenters = new ArrayList<double[][]>();
		this.m_canopyNumMissingForNumerics = new ArrayList<double[]>();

		if (data.numInstances() > 0) {
			if (!this.m_dontReplaceMissing) {
				this.m_missingValuesReplacer = new ReplaceMissingValues();
				this.m_missingValuesReplacer.setInputFormat(data);
				data = Filter.useFilter(data, this.m_missingValuesReplacer);
			}
			Random r = new Random(this.getSeed());
			for (int i = 0; i < 10; i++) {
				r.nextInt();
			}
			data.randomize(r);

			if (this.m_userT2 < 0) {
				this.setT2T1BasedOnStdDev(data);
			}
		}
		this.m_t1 = this.m_userT1 > 0 ? this.m_userT1 : -this.m_userT1 * this.m_t2;
		// if (m_t1 < m_t2) {
		// throw new Exception("T1 can't be less than T2. Computed T2 as " + m_t2
		// + " T1 is requested to be " + m_t1);
		// }

		this.m_distanceFunction.setInstances(data);

		this.m_canopies = new Instances(data, 0);
		if (data.numInstances() > 0) {
			this.m_trainingData = new Instances(data);
		}

		for (int i = 0; i < data.numInstances(); i++) {
			if (this.getDebug() && i % this.m_periodicPruningRate == 0) {
				System.err.println("Processed: " + i);
			}
			this.updateClusterer(data.instance(i));
		}

		this.updateFinished();
	}

	@Override
	public int numberOfClusters() throws Exception {
		return this.m_canopies.numInstances();
	}

	/**
	 * Set a ready-to-use missing values replacement filter
	 *
	 * @param missingReplacer the missing values replacement filter to use
	 */
	public void setMissingValuesReplacer(final Filter missingReplacer) {
		this.m_missingValuesReplacer = missingReplacer;
	}

	/**
	 * Get the canopies (cluster centers).
	 *
	 * @return the canopies
	 */
	public Instances getCanopies() {
		return this.m_canopies;
	}

	/**
	 * Set the canopies to use (replaces any learned by this clusterer already)
	 *
	 * @param canopies the canopies to use
	 */
	public void setCanopies(final Instances canopies) {
		this.m_canopies = canopies;
	}

	/**
	 * Get the canopies that each canopy (cluster center) is within T1 distance of
	 *
	 * @return a list of canopies for each cluster center
	 */
	public List<long[]> getClusterCanopyAssignments() {
		return this.m_clusterCanopies;
	}

	/**
	 * Set the canopies that each canopy (cluster center) is within T1 distance of
	 *
	 * @param clusterCanopies the list canopies for each cluster center
	 */
	public void setClusterCanopyAssignments(final List<long[]> clusterCanopies) {
		this.m_clusterCanopies = clusterCanopies;
	}

	/**
	 * Get the actual value of T2 (which may be different from the initial value
	 * if the heuristic is used)
	 *
	 * @return the actual value of T2
	 */
	public double getActualT2() {
		return this.m_t2;
	}

	/**
	 * Get the actual value of T1 (which may be different from the initial value
	 * if the heuristic is used)
	 *
	 * @return the actual value of T1
	 */
	public double getActualT1() {
		return this.m_t1;
	}

	/**
	 * Tip text for this property
	 *
	 * @return the tip text for this property
	 */
	public String t1TipText() {
		return "The T1 distance to use. Values < 0 are taken as a positive " + "multiplier for the T2 distance";
	}

	/**
	 * Set the T1 distance. Values < 0 are taken as a positive multiplier for the
	 * T2 distance - e.g. T1_actual = Math.abs(t1) * t2;
	 *
	 * @param t1 the T1 distance to use
	 */
	public void setT1(final double t1) {
		this.m_userT1 = t1;
	}

	/**
	 * Get the T1 distance. Values < 0 are taken as a positive multiplier for the
	 * T2 distance - e.g. T1_actual = Math.abs(t1) * t2;
	 *
	 * @return the T1 distance to use
	 */
	public double getT1() {
		return this.m_userT1;
	}

	/**
	 * Tip text for this property
	 *
	 * @return the tip text for this property
	 */
	public String t2TipText() {
		return "The T2 distance to use. Values < 0 indicate that this should be set using " + "a heuristic based on attribute standard deviation (note that this only" + "works when batch training)";
	}

	/**
	 * Set the T2 distance to use. Values < 0 indicate that a heuristic based on
	 * attribute standard deviation should be used to set this (note that the
	 * heuristic is only applicable when batch training).
	 *
	 * @param t2 the T2 distance to use
	 */
	public void setT2(final double t2) {
		this.m_userT2 = t2;
	}

	/**
	 * Get the T2 distance to use. Values < 0 indicate that a heuristic based on
	 * attribute standard deviation should be used to set this (note that the
	 * heuristic is only applicable when batch training).
	 *
	 * @return the T2 distance to use
	 */
	public double getT2() {
		return this.m_userT2;
	}

	/**
	 * Returns the tip text for this property.
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String numClustersTipText() {
		return "Set number of clusters. -1 means number of clusters is determined by " + "T2 distance";
	}

	@Override
	public void setNumClusters(final int numClusters) throws Exception {
		this.m_numClustersRequested = numClusters;
	}

	/**
	 * Get the number of clusters to generate
	 *
	 * @return the number of clusters to generate
	 */
	public int getNumClusters() {
		return this.m_numClustersRequested;
	}

	/**
	 * Returns the tip text for this property.
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String periodicPruningRateTipText() {
		return "How often to prune low density canopies during training";
	}

	/**
	 * Set the how often to prune low density canopies during training
	 *
	 * @param p how often (every p instances) to prune low density canopies
	 */
	public void setPeriodicPruningRate(final int p) {
		this.m_periodicPruningRate = p;
	}

	/**
	 * Get the how often to prune low density canopies during training
	 *
	 * @return how often (every p instances) to prune low density canopies
	 */
	public int getPeriodicPruningRate() {
		return this.m_periodicPruningRate;
	}

	/**
	 * Returns the tip text for this property.
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String minimumCanopyDensityTipText() {
		return "The minimum T2-based density below which a canopy will be pruned during periodic pruning";
	}

	/**
	 * Set the minimum T2-based density below which a canopy will be pruned during
	 * periodic pruning.
	 *
	 * @param dens the minimum canopy density
	 */
	public void setMinimumCanopyDensity(final double dens) {
		this.m_minClusterDensity = dens;
	}

	/**
	 * Get the minimum T2-based density below which a canopy will be pruned during
	 * periodic pruning.
	 *
	 * @return the minimum canopy density
	 */
	public double getMinimumCanopyDensity() {
		return this.m_minClusterDensity;
	}

	/**
	 * Returns the tip text for this property.
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String maxNumCandidateCanopiesToHoldInMemory() {
		return "The maximum number of candidate canopies to retain in main memory during training. " + "T2 distance and data characteristics determine how many candidate "
				+ "canopies are formed before periodic and final pruning are performed. There " + "may not be enough memory available if T2 is set too low.";
	}

	/**
	 * Set the maximum number of candidate canopies to retain in memory during
	 * training. T2 distance and data characteristics determine how many candidate
	 * canopies are formed before periodic and final pruning are performed. There
	 * may not be enough memory available if T2 is set too low.
	 *
	 * @param max the maximum number of candidate canopies to retain in memory
	 *          during training
	 */
	public void setMaxNumCandidateCanopiesToHoldInMemory(final int max) {
		this.m_maxCanopyCandidates = max;
	}

	/**
	 * Get the maximum number of candidate canopies to retain in memory during
	 * training. T2 distance and data characteristics determine how many candidate
	 * canopies are formed before periodic and final pruning are performed. There
	 * may not be enough memory available if T2 is set too low.
	 *
	 * @return the maximum number of candidate canopies to retain in memory during
	 *         training
	 */
	public int getMaxNumCandidateCanopiesToHoldInMemory() {
		return this.m_maxCanopyCandidates;
	}

	/**
	 * Returns the tip text for this property.
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String dontReplaceMissingValuesTipText() {
		return "Replace missing values globally with mean/mode.";
	}

	/**
	 * Sets whether missing values are to be replaced.
	 *
	 * @param r true if missing values are to be replaced
	 */
	public void setDontReplaceMissingValues(final boolean r) {
		this.m_dontReplaceMissing = r;
	}

	/**
	 * Gets whether missing values are to be replaced.
	 *
	 * @return true if missing values are to be replaced
	 */
	public boolean getDontReplaceMissingValues() {
		return this.m_dontReplaceMissing;
	}

	public static String printSingleAssignment(final long[] assignments) {
		StringBuilder temp = new StringBuilder();

		boolean first = true;
		temp.append(" <");
		for (int j = 0; j < assignments.length; j++) {
			long block = assignments[j];
			int offset = j * 64;

			for (int k = 0; k < 64; k++) {
				long mask = 1L << k;

				if ((mask & block) != 0L) {
					temp.append("" + (!first ? "," : "") + (offset + k));
					if (first) {
						first = false;
					}
				}
			}
		}

		temp.append(">");
		return temp.toString();
	}

	/**
	 * Print the supplied instances and their canopies
	 *
	 * @param dataPoints the instances to print
	 * @param canopyAssignments the canopy assignments, one assignment array for
	 *          each instance
	 * @return a string containing the printed assignments
	 */
	public static String printCanopyAssignments(final Instances dataPoints, final List<long[]> canopyAssignments) {
		StringBuilder temp = new StringBuilder();

		for (int i = 0; i < dataPoints.size(); i++) {
			temp.append("Cluster " + i + ": ");
			temp.append(dataPoints.instance(i));
			if (canopyAssignments != null && canopyAssignments.size() == dataPoints.size()) {
				long[] assignments = canopyAssignments.get(i);
				temp.append(printSingleAssignment(assignments));
			}
			temp.append("\n");
		}

		return temp.toString();
	}

	/**
	 * Return a textual description of this clusterer
	 *
	 * @param header true if the header should be printed
	 * @return a string describing the result of the clustering
	 */
	public String toString(final boolean header) {
		StringBuffer temp = new StringBuffer();

		if (this.m_canopies == null) {
			return "No clusterer built yet";
		}

		if (header) {
			temp.append("\nCanopy clustering\n=================\n");
			temp.append("\nNumber of canopies (cluster centers) found: " + this.m_canopies.numInstances());
		}

		temp.append("\nT2 radius: " + String.format("%-10.3f", this.m_t2));
		temp.append("\nT1 radius: " + String.format("%-10.3f", this.m_t1));
		temp.append("\n\n");

		temp.append(printCanopyAssignments(this.m_canopies, this.m_clusterCanopies));

		temp.append("\n");

		return temp.toString();
	}

	@Override
	public String toString() {
		return this.toString(true);
	}

	/**
	 * Save memory
	 */
	public void cleanUp() {
		this.m_canopyNumMissingForNumerics = null;
		this.m_canopyT2Density = null;
		this.m_canopyCenters = null;
	}

	/**
	 * Aggregate the canopies from a list of Canopy clusterers together into one
	 * final model.
	 *
	 * @param canopies the list of Canopy clusterers to aggregate
	 * @param aggregationT1 the T1 distance to use for the aggregated classifier
	 * @param aggregationT2 the T2 distance to use when aggregating canopies
	 * @param finalDistanceFunction the distance function to use with the final
	 *          Canopy clusterer
	 * @param missingValuesReplacer the missing value replacement filter to use
	 *          with the final clusterer (can be null for no missing value
	 *          replacement)
	 * @param finalNumCanopies the final number of canopies
	 * @return a Canopy clusterer that aggregates all the canopies
	 * @throws InterruptedException
	 */
	public static Canopy aggregateCanopies(final List<Canopy> canopies, final double aggregationT1, final double aggregationT2, final NormalizableDistance finalDistanceFunction, final Filter missingValuesReplacer,
			final int finalNumCanopies) throws InterruptedException {

		Instances collectedCanopies = new Instances(canopies.get(0).getCanopies(), 0);
		Instances finalCanopies = new Instances(collectedCanopies, 0);

		List<double[][]> finalCenters = new ArrayList<double[][]>();
		List<double[]> finalMissingNumerics = new ArrayList<double[]>();
		List<double[]> finalT2Densities = new ArrayList<double[]>();
		List<Instance> finalCanopiesList = new ArrayList<Instance>();
		List<double[][]> centersForEachCanopy = new ArrayList<double[][]>();
		List<double[]> numMissingNumericsForEachCanopy = new ArrayList<double[]>();

		for (Canopy c : canopies) {
			Instances tempC = c.getCanopies();
			// System.err.println("A canopy clusterer:\n " + c.toString());
			for (int i = 0; i < tempC.numInstances(); i++) {
				collectedCanopies.add(tempC.instance(i));
				centersForEachCanopy.add(c.m_canopyCenters.get(i));
				numMissingNumericsForEachCanopy.add(c.m_canopyNumMissingForNumerics.get(i));
			}
		}

		for (int i = 0; i < collectedCanopies.numInstances(); i++) {
			boolean addPoint = true;
			Instance candidate = collectedCanopies.instance(i);
			double[][] candidateCenter = centersForEachCanopy.get(i);
			double[] candidateMissingNumerics = numMissingNumericsForEachCanopy.get(i);

			for (int j = 0; j < finalCanopiesList.size(); j++) {
				Instance fc = finalCanopiesList.get(j);

				if (finalDistanceFunction.distance(candidate, fc) < aggregationT2) {
					addPoint = false;

					// now absorb candidate into fc
					double[][] center = finalCenters.get(j);
					double[] missingNumerics = finalMissingNumerics.get(j);
					// double newDensity = fc.weight() + candidate.weight();
					finalT2Densities.get(j)[0] += candidate.weight();

					for (int k = 0; k < candidate.numAttributes(); k++) {
						missingNumerics[k] += candidateMissingNumerics[k];
						for (int l = 0; l < center[k].length; l++) {
							center[k][l] += candidateCenter[k][l];
						}
					}

					break;
				}
			}

			if (addPoint) {
				finalCanopiesList.add(candidate);
				finalCanopies.add(candidate);
				finalCenters.add(candidateCenter);
				finalMissingNumerics.add(candidateMissingNumerics);
				double[] dens = new double[1];
				dens[0] = candidate.weight();
				finalT2Densities.add(dens);
			}
		}

		// now construct a new Canopy encapsulating the final set of canopies
		// System.err.println(finalCanopies);
		Canopy finalC = new Canopy();
		finalC.setCanopies(finalCanopies);
		finalC.setMissingValuesReplacer(missingValuesReplacer);
		finalC.m_distanceFunction = finalDistanceFunction;
		finalC.m_canopyCenters = finalCenters;
		finalC.m_canopyNumMissingForNumerics = finalMissingNumerics;
		finalC.m_canopyT2Density = finalT2Densities;
		finalC.m_t2 = aggregationT2;
		finalC.m_t1 = aggregationT1;
		try {
			finalC.setNumClusters(finalNumCanopies);
		} catch (Exception e) {
			// can safely ignore as Canopy does not generate an exception
		}

		finalC.updateFinished();

		return finalC;
	}

	public static void main(final String[] args) {
		runClusterer(new Canopy(), args);
	}

}
