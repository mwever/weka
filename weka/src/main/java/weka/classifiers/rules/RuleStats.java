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
 *    RuleStats.java
 *    Copyright (C) 2001-2012 University of Waikato, Hamilton, New Zealand
 */

package weka.classifiers.rules;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * This class implements the statistics functions used in the propositional rule learner, from the simpler ones like count of true/false positive/negatives, filter data based on the ruleset, etc. to the more sophisticated ones such as MDL
 * calculation and rule variants generation for each rule in the ruleset.
 * <p>
 *
 * Obviously the statistics functions listed above need the specific data and the specific ruleset, which are given in order to instantiate an object of this class.
 * <p>
 *
 * @author Xin Xu (xx5@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class RuleStats implements Serializable, RevisionHandler {

	/** for serialization */
	static final long serialVersionUID = -5708153367675298624L;

	/** The data on which the stats calculation is based */
	private Instances m_Data;

	/** The specific ruleset in question */
	private ArrayList<Rule> m_Ruleset;

	/** The simple stats of each rule */
	private ArrayList<double[]> m_SimpleStats;

	/** The set of instances filtered by the ruleset */
	private ArrayList<Instances[]> m_Filtered;

	/**
	 * The total number of possible conditions that could appear in a rule
	 */
	private double m_Total;

	/** The redundancy factor in theory description length */
	private static double REDUNDANCY_FACTOR = 0.5;

	/** The theory weight in the MDL calculation */
	private double MDL_THEORY_WEIGHT = 1.0;

	/** The class distributions predicted by each rule */
	private ArrayList<double[]> m_Distributions;

	/** Default constructor */
	public RuleStats() {
		this.m_Data = null;
		this.m_Ruleset = null;
		this.m_SimpleStats = null;
		this.m_Filtered = null;
		this.m_Distributions = null;
		this.m_Total = -1;
	}

	/**
	 * Constructor that provides ruleset and data
	 *
	 * @param data
	 *            the data
	 * @param rules
	 *            the ruleset
	 */
	public RuleStats(final Instances data, final ArrayList<Rule> rules) {
		this();
		this.m_Data = data;
		this.m_Ruleset = rules;
	}

	/**
	 * Frees up memory after classifier has been built.
	 */
	public void cleanUp() {
		this.m_Data = null;
		this.m_Filtered = null;
	}

	/**
	 * Set the number of all conditions that could appear in a rule in this RuleStats object, if the number set is smaller than 0 (typically -1), then it calcualtes based on the data store
	 *
	 * @param total
	 *            the set number
	 * @throws InterruptedException
	 */
	public void setNumAllConds(final double total) throws InterruptedException {
		if (total < 0) {
			this.m_Total = numAllConditions(this.m_Data);
		} else {
			this.m_Total = total;
		}
	}

	/**
	 * Set the data of the stats, overwriting the old one if any
	 *
	 * @param data
	 *            the data to be set
	 */
	public void setData(final Instances data) {
		this.m_Data = data;
	}

	/**
	 * Get the data of the stats
	 *
	 * @return the data
	 */
	public Instances getData() {
		return this.m_Data;
	}

	/**
	 * Set the ruleset of the stats, overwriting the old one if any
	 *
	 * @param rules
	 *            the set of rules to be set
	 */
	public void setRuleset(final ArrayList<Rule> rules) {
		this.m_Ruleset = rules;
	}

	/**
	 * Get the ruleset of the stats
	 *
	 * @return the set of rules
	 */
	public ArrayList<Rule> getRuleset() {
		return this.m_Ruleset;
	}

	/**
	 * Get the size of the ruleset in the stats
	 *
	 * @return the size of ruleset
	 */
	public int getRulesetSize() {
		return this.m_Ruleset.size();
	}

	/**
	 * Get the simple stats of one rule, including 6 parameters: 0: coverage; 1:uncoverage; 2: true positive; 3: true negatives; 4: false positives; 5: false negatives
	 *
	 * @param index
	 *            the index of the rule
	 * @return the stats
	 */
	public double[] getSimpleStats(final int index) {
		if ((this.m_SimpleStats != null) && (index < this.m_SimpleStats.size())) {
			return this.m_SimpleStats.get(index);
		}

		return null;
	}

	/**
	 * Get the data after filtering the given rule
	 *
	 * @param index
	 *            the index of the rule
	 * @return the data covered and uncovered by the rule
	 */
	public Instances[] getFiltered(final int index) {

		if ((this.m_Filtered != null) && (index < this.m_Filtered.size())) {
			return this.m_Filtered.get(index);
		}

		return null;
	}

	/**
	 * Get the class distribution predicted by the rule in given position
	 *
	 * @param index
	 *            the position index of the rule
	 * @return the class distributions
	 */
	public double[] getDistributions(final int index) {

		if ((this.m_Distributions != null) && (index < this.m_Distributions.size())) {
			return this.m_Distributions.get(index);
		}

		return null;
	}

	/**
	 * Set the weight of theory in MDL calcualtion
	 *
	 * @param weight
	 *            the weight to be set
	 */
	public void setMDLTheoryWeight(final double weight) {
		this.MDL_THEORY_WEIGHT = weight;
	}

	/**
	 * Compute the number of all possible conditions that could appear in a rule of a given data. For nominal attributes, it's the number of values that could appear; for numeric attributes, it's the number of values * 2, i.e. <= and >= are
	 * counted as different possible conditions.
	 *
	 * @param data
	 *            the given data
	 * @return number of all conditions of the data
	 * @throws InterruptedException
	 */
	public static double numAllConditions(final Instances data) throws InterruptedException {
		double total = 0;
		Enumeration<Attribute> attEnum = data.enumerateAttributes();
		while (attEnum.hasMoreElements()) {
			Attribute att = attEnum.nextElement();
			if (att.isNominal()) {
				total += att.numValues();
			} else {
				total += 2.0 * data.numDistinctValues(att);
			}
		}
		return total;
	}

	/**
	 * Filter the data according to the ruleset and compute the basic stats: coverage/uncoverage, true/false positive/negatives of each rule
	 */
	public void countData() {
		if ((this.m_Filtered != null) || (this.m_Ruleset == null) || (this.m_Data == null)) {
			return;
		}

		int size = this.m_Ruleset.size();
		this.m_Filtered = new ArrayList<Instances[]>(size);
		this.m_SimpleStats = new ArrayList<double[]>(size);
		this.m_Distributions = new ArrayList<double[]>(size);
		Instances data = new Instances(this.m_Data);

		for (int i = 0; i < size; i++) {
			double[] stats = new double[6]; // 6 statistics parameters
			double[] classCounts = new double[this.m_Data.classAttribute().numValues()];
			Instances[] filtered = this.computeSimpleStats(i, data, stats, classCounts);
			this.m_Filtered.add(filtered);
			this.m_SimpleStats.add(stats);
			this.m_Distributions.add(classCounts);
			data = filtered[1]; // Data not covered
		}
	}

	/**
	 * Count data from the position index in the ruleset assuming that given data are not covered by the rules in position 0...(index-1), and the statistics of these rules are provided.<br>
	 * This procedure is typically useful when a temporary object of RuleStats is constructed in order to efficiently calculate the relative DL of rule in position index, thus all other stuff is not needed.
	 *
	 * @param index
	 *            the given position
	 * @param uncovered
	 *            the data not covered by rules before index
	 * @param prevRuleStats
	 *            the provided stats of previous rules
	 */
	public void countData(final int index, final Instances uncovered, final double[][] prevRuleStats) {
		if ((this.m_Filtered != null) || (this.m_Ruleset == null)) {
			return;
		}

		int size = this.m_Ruleset.size();
		this.m_Filtered = new ArrayList<Instances[]>(size);
		this.m_SimpleStats = new ArrayList<double[]>(size);
		Instances[] data = new Instances[2];
		data[1] = uncovered;

		for (int i = 0; i < index; i++) {
			this.m_SimpleStats.add(prevRuleStats[i]);
			if (i + 1 == index) {
				this.m_Filtered.add(data);
			} else {
				this.m_Filtered.add(new Instances[0]); // Stuff sth.
			}
		}

		for (int j = index; j < size; j++) {
			double[] stats = new double[6]; // 6 statistics parameters
			Instances[] filtered = this.computeSimpleStats(j, data[1], stats, null);
			this.m_Filtered.add(filtered);
			this.m_SimpleStats.add(stats);
			data = filtered; // Data not covered
		}
	}

	/**
	 * Find all the instances in the dataset covered/not covered by the rule in given index, and the correponding simple statistics and predicted class distributions are stored in the given double array, which can be obtained by
	 * getSimpleStats() and getDistributions().<br>
	 *
	 * @param index
	 *            the given index, assuming correct
	 * @param insts
	 *            the dataset to be covered by the rule
	 * @param stats
	 *            the given double array to hold stats, side-effected
	 * @param dist
	 *            the given array to hold class distributions, side-effected if null, the distribution is not necessary
	 * @return the instances covered and not covered by the rule
	 */
	private Instances[] computeSimpleStats(final int index, final Instances insts, final double[] stats, final double[] dist) {
		Rule rule = this.m_Ruleset.get(index);

		Instances[] data = new Instances[2];
		data[0] = new Instances(insts, insts.numInstances());
		data[1] = new Instances(insts, insts.numInstances());

		for (int i = 0; i < insts.numInstances(); i++) {
			Instance datum = insts.instance(i);
			double weight = datum.weight();
			if (rule.covers(datum)) {
				data[0].add(datum); // Covered by this rule
				stats[0] += weight; // Coverage
				if ((int) datum.classValue() == (int) rule.getConsequent()) {
					stats[2] += weight; // True positives
				} else {
					stats[4] += weight; // False positives
				}
				if (dist != null) {
					dist[(int) datum.classValue()] += weight;
				}
			} else {
				data[1].add(datum); // Not covered by this rule
				stats[1] += weight;
				if ((int) datum.classValue() != (int) rule.getConsequent()) {
					stats[3] += weight; // True negatives
				} else {
					stats[5] += weight; // False negatives
				}
			}
		}

		return data;
	}

	/**
	 * Add a rule to the ruleset and update the stats
	 *
	 * @param lastRule
	 *            the rule to be added
	 */
	public void addAndUpdate(final Rule lastRule) {
		if (this.m_Ruleset == null) {
			this.m_Ruleset = new ArrayList<Rule>();
		}
		this.m_Ruleset.add(lastRule);

		Instances data = (this.m_Filtered == null) ? this.m_Data : (this.m_Filtered.get(this.m_Filtered.size() - 1))[1];
		double[] stats = new double[6];
		double[] classCounts = new double[this.m_Data.classAttribute().numValues()];
		Instances[] filtered = this.computeSimpleStats(this.m_Ruleset.size() - 1, data, stats, classCounts);

		if (this.m_Filtered == null) {
			this.m_Filtered = new ArrayList<Instances[]>();
		}
		this.m_Filtered.add(filtered);

		if (this.m_SimpleStats == null) {
			this.m_SimpleStats = new ArrayList<double[]>();
		}
		this.m_SimpleStats.add(stats);

		if (this.m_Distributions == null) {
			this.m_Distributions = new ArrayList<double[]>();
		}
		this.m_Distributions.add(classCounts);
	}

	/**
	 * Subset description length: <br>
	 * S(t,k,p) = -k*log2(p)-(n-k)log2(1-p)
	 *
	 * Details see Quilan: "MDL and categorical theories (Continued)",ML95
	 *
	 * @param t
	 *            the number of elements in a known set
	 * @param k
	 *            the number of elements in a subset
	 * @param p
	 *            the expected proportion of subset known by recipient
	 * @return the subset description length
	 */
	public static double subsetDL(final double t, final double k, final double p) {
		double rt = Utils.gr(p, 0.0) ? (-k * Utils.log2(p)) : 0.0;
		rt -= (t - k) * Utils.log2(1 - p);
		return rt;
	}

	/**
	 * The description length of the theory for a given rule. Computed as:<br>
	 * 0.5* [||k||+ S(t, k, k/t)]<br>
	 * where k is the number of antecedents of the rule; t is the total possible antecedents that could appear in a rule; ||K|| is the universal prior for k , log2*(k) and S(t,k,p) = -k*log2(p)-(n-k)log2(1-p) is the subset encoding length.
	 * <p>
	 *
	 * Details see Quilan: "MDL and categorical theories (Continued)",ML95
	 *
	 * @param index
	 *            the index of the given rule (assuming correct)
	 * @return the theory DL, weighted if weight != 1.0
	 */
	public double theoryDL(final int index) {

		double k = this.m_Ruleset.get(index).size();

		if (k == 0) {
			return 0.0;
		}

		double tdl = Utils.log2(k);
		if (k > 1) {
			tdl += 2.0 * Utils.log2(tdl); // of log2 star
		}
		tdl += subsetDL(this.m_Total, k, k / this.m_Total);
		return this.MDL_THEORY_WEIGHT * REDUNDANCY_FACTOR * tdl;
	}

	/**
	 * The description length of data given the parameters of the data based on the ruleset.
	 * <p>
	 * Details see Quinlan: "MDL and categorical theories (Continued)",ML95
	 * <p>
	 *
	 * @param expFPOverErr
	 *            expected FP/(FP+FN)
	 * @param cover
	 *            coverage
	 * @param uncover
	 *            uncoverage
	 * @param fp
	 *            False Positive
	 * @param fn
	 *            False Negative
	 * @return the description length
	 */
	public static double dataDL(final double expFPOverErr, final double cover, final double uncover, final double fp, final double fn) {
		double totalBits = Utils.log2(cover + uncover + 1.0); // how many data?
		double coverBits, uncoverBits; // What's the error?
		double expErr; // Expected FP or FN

		if (Utils.gr(cover, uncover)) {
			expErr = expFPOverErr * (fp + fn);
			coverBits = subsetDL(cover, fp, expErr / cover);
			uncoverBits = Utils.gr(uncover, 0.0) ? subsetDL(uncover, fn, fn / uncover) : 0.0;
		} else {
			expErr = (1.0 - expFPOverErr) * (fp + fn);
			coverBits = Utils.gr(cover, 0.0) ? subsetDL(cover, fp, fp / cover) : 0.0;
			uncoverBits = subsetDL(uncover, fn, expErr / uncover);
		}

		/*
		 * System.err.println("!!!cover: " + cover + "|uncover" + uncover +
		 * "|coverBits: "+coverBits+"|uncBits: "+ uncoverBits+
		 * "|FPRate: "+expFPOverErr + "|expErr: "+expErr+
		 * "|fp: "+fp+"|fn: "+fn+"|total: "+totalBits);
		 */
		return (totalBits + coverBits + uncoverBits);
	}

	/**
	 * Calculate the potential to decrease DL of the ruleset, i.e. the possible DL that could be decreased by deleting the rule whose index and simple statstics are given. If there's no potentials (i.e. smOrEq 0 && error rate < 0.5), it
	 * returns NaN.
	 * <p>
	 *
	 * The way this procedure does is copied from original RIPPER implementation and is quite bizzare because it does not update the following rules' stats recursively any more when testing each rule, which means it assumes after deletion
	 * no data covered by the following rules (or regards the deleted rule as the last rule). Reasonable assumption?
	 * <p>
	 *
	 * @param index
	 *            the index of the rule in m_Ruleset to be deleted
	 * @param expFPOverErr
	 *            expected FP/(FP+FN)
	 * @param rulesetStat
	 *            the simple statistics of the ruleset, updated if the rule should be deleted
	 * @param ruleStat
	 *            the simple statistics of the rule to be deleted
	 * @param checkErr
	 *            whether check if error rate >= 0.5
	 * @return the potential DL that could be decreased
	 */
	public double potential(final int index, final double expFPOverErr, final double[] rulesetStat, final double[] ruleStat, final boolean checkErr) {
		// Restore the stats if deleted
		double pcov = rulesetStat[0] - ruleStat[0];
		double puncov = rulesetStat[1] + ruleStat[0];
		double pfp = rulesetStat[4] - ruleStat[4];
		double pfn = rulesetStat[5] + ruleStat[2];

		double dataDLWith = dataDL(expFPOverErr, rulesetStat[0], rulesetStat[1], rulesetStat[4], rulesetStat[5]);
		double theoryDLWith = this.theoryDL(index);
		double dataDLWithout = dataDL(expFPOverErr, pcov, puncov, pfp, pfn);

		double potential = dataDLWith + theoryDLWith - dataDLWithout;
		double err = ruleStat[4] / ruleStat[0];
		boolean overErr = Utils.grOrEq(err, 0.5);
		if (!checkErr) {
			overErr = false;
		}

		if (Utils.grOrEq(potential, 0.0) || overErr) {
			// If deleted, update ruleset stats. Other stats do not matter
			rulesetStat[0] = pcov;
			rulesetStat[1] = puncov;
			rulesetStat[4] = pfp;
			rulesetStat[5] = pfn;
			return potential;
		} else {
			return Double.NaN;
		}
	}

	/**
	 * Compute the minimal data description length of the ruleset if the rule in the given position is deleted.<br>
	 * The min_data_DL_if_deleted = data_DL_if_deleted - potential
	 *
	 * @param index
	 *            the index of the rule in question
	 * @param expFPRate
	 *            expected FP/(FP+FN), used in dataDL calculation
	 * @param checkErr
	 *            whether check if error rate >= 0.5
	 * @return the minDataDL
	 */
	public double minDataDLIfDeleted(final int index, final double expFPRate, final boolean checkErr) {
		double[] rulesetStat = new double[6]; // Stats of ruleset if deleted
		int more = this.m_Ruleset.size() - 1 - index; // How many rules after?
		ArrayList<double[]> indexPlus = new ArrayList<double[]>(more); // Their
																		// stats

		// 0...(index-1) are OK
		for (int j = 0; j < index; j++) {
			// Covered stats are cumulative
			rulesetStat[0] += this.m_SimpleStats.get(j)[0];
			rulesetStat[2] += this.m_SimpleStats.get(j)[2];
			rulesetStat[4] += this.m_SimpleStats.get(j)[4];
		}

		// Recount data from index+1
		Instances data = (index == 0) ? this.m_Data : this.m_Filtered.get(index - 1)[1];

		for (int j = (index + 1); j < this.m_Ruleset.size(); j++) {
			double[] stats = new double[6];
			Instances[] split = this.computeSimpleStats(j, data, stats, null);
			indexPlus.add(stats);
			rulesetStat[0] += stats[0];
			rulesetStat[2] += stats[2];
			rulesetStat[4] += stats[4];
			data = split[1];
		}
		// Uncovered stats are those of the last rule
		if (more > 0) {
			rulesetStat[1] = indexPlus.get(indexPlus.size() - 1)[1];
			rulesetStat[3] = indexPlus.get(indexPlus.size() - 1)[3];
			rulesetStat[5] = indexPlus.get(indexPlus.size() - 1)[5];
		} else if (index > 0) {
			rulesetStat[1] = this.m_SimpleStats.get(index - 1)[1];
			rulesetStat[3] = this.m_SimpleStats.get(index - 1)[3];
			rulesetStat[5] = this.m_SimpleStats.get(index - 1)[5];
		} else { // Null coverage
			rulesetStat[1] = this.m_SimpleStats.get(0)[0] + this.m_SimpleStats.get(0)[1];
			rulesetStat[3] = this.m_SimpleStats.get(0)[3] + this.m_SimpleStats.get(0)[4];
			rulesetStat[5] = this.m_SimpleStats.get(0)[2] + this.m_SimpleStats.get(0)[5];
		}

		// Potential
		double potential = 0;
		for (int k = index + 1; k < this.m_Ruleset.size(); k++) {
			double[] ruleStat = indexPlus.get(k - index - 1);
			double ifDeleted = this.potential(k, expFPRate, rulesetStat, ruleStat, checkErr);
			if (!Double.isNaN(ifDeleted)) {
				potential += ifDeleted;
			}
		}

		// Data DL of the ruleset without the rule
		// Note that ruleset stats has already been updated to reflect
		// deletion if any potential
		double dataDLWithout = dataDL(expFPRate, rulesetStat[0], rulesetStat[1], rulesetStat[4], rulesetStat[5]);
		// Why subtract potential again? To reflect change of theory DL??
		return (dataDLWithout - potential);
	}

	/**
	 * Compute the minimal data description length of the ruleset if the rule in the given position is NOT deleted.<br>
	 * The min_data_DL_if_n_deleted = data_DL_if_n_deleted - potential
	 *
	 * @param index
	 *            the index of the rule in question
	 * @param expFPRate
	 *            expected FP/(FP+FN), used in dataDL calculation
	 * @param checkErr
	 *            whether check if error rate >= 0.5
	 * @return the minDataDL
	 */
	public double minDataDLIfExists(final int index, final double expFPRate, final boolean checkErr) {
		double[] rulesetStat = new double[6]; // Stats of ruleset if rule exists
		for (int j = 0; j < this.m_SimpleStats.size(); j++) {
			// Covered stats are cumulative
			rulesetStat[0] += this.m_SimpleStats.get(j)[0];
			rulesetStat[2] += this.m_SimpleStats.get(j)[2];
			rulesetStat[4] += this.m_SimpleStats.get(j)[4];
			if (j == this.m_SimpleStats.size() - 1) { // Last rule
				rulesetStat[1] = this.m_SimpleStats.get(j)[1];
				rulesetStat[3] = this.m_SimpleStats.get(j)[3];
				rulesetStat[5] = this.m_SimpleStats.get(j)[5];
			}
		}

		// Potential
		double potential = 0;
		for (int k = index + 1; k < this.m_SimpleStats.size(); k++) {
			double[] ruleStat = this.getSimpleStats(k);
			double ifDeleted = this.potential(k, expFPRate, rulesetStat, ruleStat, checkErr);
			if (!Double.isNaN(ifDeleted)) {
				potential += ifDeleted;
			}
		}

		// Data DL of the ruleset without the rule
		// Note that ruleset stats has already been updated to reflect deletion
		// if any potential
		double dataDLWith = dataDL(expFPRate, rulesetStat[0], rulesetStat[1], rulesetStat[4], rulesetStat[5]);
		return (dataDLWith - potential);
	}

	/**
	 * The description length (DL) of the ruleset relative to if the rule in the given position is deleted, which is obtained by: <br>
	 * MDL if the rule exists - MDL if the rule does not exist <br>
	 * Note the minimal possible DL of the ruleset is calculated(i.e. some other rules may also be deleted) instead of the DL of the current ruleset.
	 * <p>
	 *
	 * @param index
	 *            the given position of the rule in question (assuming correct)
	 * @param expFPRate
	 *            expected FP/(FP+FN), used in dataDL calculation
	 * @param checkErr
	 *            whether check if error rate >= 0.5
	 * @return the relative DL
	 */
	public double relativeDL(final int index, final double expFPRate, final boolean checkErr) {

		return (this.minDataDLIfExists(index, expFPRate, checkErr) + this.theoryDL(index) - this.minDataDLIfDeleted(index, expFPRate, checkErr));
	}

	/**
	 * Try to reduce the DL of the ruleset by testing removing the rules one by one in reverse order and update all the stats
	 *
	 * @param expFPRate
	 *            expected FP/(FP+FN), used in dataDL calculation
	 * @param checkErr
	 *            whether check if error rate >= 0.5
	 */
	public void reduceDL(final double expFPRate, final boolean checkErr) {

		boolean needUpdate = false;
		double[] rulesetStat = new double[6];
		for (int j = 0; j < this.m_SimpleStats.size(); j++) {
			// Covered stats are cumulative
			rulesetStat[0] += this.m_SimpleStats.get(j)[0];
			rulesetStat[2] += this.m_SimpleStats.get(j)[2];
			rulesetStat[4] += this.m_SimpleStats.get(j)[4];
			if (j == this.m_SimpleStats.size() - 1) { // Last rule
				rulesetStat[1] = this.m_SimpleStats.get(j)[1];
				rulesetStat[3] = this.m_SimpleStats.get(j)[3];
				rulesetStat[5] = this.m_SimpleStats.get(j)[5];
			}
		}

		// Potential
		for (int k = this.m_SimpleStats.size() - 1; k >= 0; k--) {

			double[] ruleStat = this.m_SimpleStats.get(k);

			// rulesetStat updated
			double ifDeleted = this.potential(k, expFPRate, rulesetStat, ruleStat, checkErr);
			if (!Double.isNaN(ifDeleted)) {
				/*
				 * System.err.println("!!!deleted ("+k+"): save "+ifDeleted
				 * +" | "+rulesetStat[0] +" | "+rulesetStat[1] +" | "+rulesetStat[4]
				 * +" | "+rulesetStat[5]);
				 */

				if (k == (this.m_SimpleStats.size() - 1)) {
					this.removeLast();
				} else {
					this.m_Ruleset.remove(k);
					needUpdate = true;
				}
			}
		}

		if (needUpdate) {
			this.m_Filtered = null;
			this.m_SimpleStats = null;
			this.countData();
		}
	}

	/**
	 * Remove the last rule in the ruleset as well as it's stats. It might be useful when the last rule was added for testing purpose and then the test failed
	 */
	public void removeLast() {
		int last = this.m_Ruleset.size() - 1;
		this.m_Ruleset.remove(last);
		this.m_Filtered.remove(last);
		this.m_SimpleStats.remove(last);
		if (this.m_Distributions != null) {
			this.m_Distributions.remove(last);
		}
	}

	/**
	 * Static utility function to count the data covered by the rules after the given index in the given rules, and then remove them. It returns the data not covered by the successive rules.
	 *
	 * @param data
	 *            the data to be processed
	 * @param rules
	 *            the ruleset
	 * @param index
	 *            the given index
	 * @return the data after processing
	 */
	public static Instances rmCoveredBySuccessives(final Instances data, final ArrayList<Rule> rules, final int index) {
		Instances rt = new Instances(data, 0);

		for (int i = 0; i < data.numInstances(); i++) {
			Instance datum = data.instance(i);
			boolean covered = false;

			for (int j = index + 1; j < rules.size(); j++) {
				Rule rule = rules.get(j);
				if (rule.covers(datum)) {
					covered = true;
					break;
				}
			}

			if (!covered) {
				rt.add(datum);
			}
		}
		return rt;
	}

	/**
	 * Stratify the given data into the given number of bags based on the class values. It differs from the <code>Instances.stratify(int fold)</code> that before stratification it sorts the instances according to the class order in the
	 * header file. It assumes no missing values in the class.
	 *
	 * @param data
	 *            the given data
	 * @param folds
	 *            the given number of folds
	 * @param rand
	 *            the random object used to randomize the instances
	 * @return the stratified instances
	 * @throws InterruptedException
	 */
	public static final Instances stratify(final Instances data, final int folds, final Random rand) throws InterruptedException {
		if (!data.classAttribute().isNominal()) {
			return data;
		}

		Instances result = new Instances(data, 0);
		Instances[] bagsByClasses = new Instances[data.numClasses()];

		for (int i = 0; i < bagsByClasses.length; i++) {
			bagsByClasses[i] = new Instances(data, 0);
		}

		// Sort by class
		for (int j = 0; j < data.numInstances(); j++) {
			Instance datum = data.instance(j);
			bagsByClasses[(int) datum.classValue()].add(datum);
		}

		// Randomize each class
		for (Instances bagsByClasse : bagsByClasses) {
			if (Thread.interrupted()) {
				throw new InterruptedException("Killed WEKA!");
			}
			bagsByClasse.randomize(rand);
		}

		for (int k = 0; k < folds; k++) {
			int offset = k, bag = 0;
			oneFold: while (true) {
				while (offset >= bagsByClasses[bag].numInstances()) {
					offset -= bagsByClasses[bag].numInstances();
					if (++bag >= bagsByClasses.length) {
						break oneFold;
					}
				}

				result.add(bagsByClasses[bag].instance(offset));
				offset += folds;
			}
		}

		return result;
	}

	/**
	 * Compute the combined DL of the ruleset in this class, i.e. theory DL and data DL. Note this procedure computes the combined DL according to the current status of the ruleset in this class
	 *
	 * @param expFPRate
	 *            expected FP/(FP+FN), used in dataDL calculation
	 * @param predicted
	 *            the default classification if ruleset covers null
	 * @return the combined class
	 */
	public double combinedDL(final double expFPRate, final double predicted) {
		double rt = 0;

		if (this.getRulesetSize() > 0) {
			double[] stats = this.m_SimpleStats.get(this.m_SimpleStats.size() - 1);
			for (int j = this.getRulesetSize() - 2; j >= 0; j--) {
				stats[0] += this.getSimpleStats(j)[0];
				stats[2] += this.getSimpleStats(j)[2];
				stats[4] += this.getSimpleStats(j)[4];
			}
			rt += dataDL(expFPRate, stats[0], stats[1], stats[4], stats[5]); // Data
																				// DL
		} else { // Null coverage ruleset
			double fn = 0.0;
			for (int j = 0; j < this.m_Data.numInstances(); j++) {
				if ((int) this.m_Data.instance(j).classValue() == (int) predicted) {
					fn += this.m_Data.instance(j).weight();
				}
			}
			rt += dataDL(expFPRate, 0.0, this.m_Data.sumOfWeights(), 0.0, fn);
		}

		for (int i = 0; i < this.getRulesetSize(); i++) {
			rt += this.theoryDL(i);
		}

		return rt;
	}

	/**
	 * Patition the data into 2, first of which has (numFolds-1)/numFolds of the data and the second has 1/numFolds of the data
	 *
	 *
	 * @param data
	 *            the given data
	 * @param numFolds
	 *            the given number of folds
	 * @return the patitioned instances
	 */
	public static final Instances[] partition(final Instances data, final int numFolds) {
		Instances[] rt = new Instances[2];
		int splits = data.numInstances() * (numFolds - 1) / numFolds;

		rt[0] = new Instances(data, 0, splits);
		rt[1] = new Instances(data, splits, data.numInstances() - splits);

		return rt;
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
