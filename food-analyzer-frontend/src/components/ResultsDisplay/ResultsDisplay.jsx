import React from 'react';
import { motion } from 'framer-motion';
import NutritionChart from '../NutritionChart/NutritionChart';
import { capitalizeWords, formatNutritionValue, calculateCalories, getConfidenceColor } from '../../utils/helpers';
import './ResultsDisplay.css';

const ResultsDisplay = ({ analysis, onNewAnalysis }) => {
    if (!analysis) return null;

    const { food_type, confidence, nutrition, success, message } = analysis;

    const isLowAccuracy = (success === false && message === 'Accuracy less than 70%') || confidence < 0.7;

    const hasNutritionError = food_type && !nutrition;

    const isFullySuccessful = food_type && nutrition && confidence >= 0.7;

    const renderLowAccuracyWarning = () => (
        <motion.div
            className="warning-container warning-low-accuracy card"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4 }}
        >
            <div className="warning-icon">⚠️</div>
            <div className="warning-content">
                <h3>Низька точність розпізнавання</h3>
                <p>
                    Система розпізнала продукт з точністю менше 70%. Результат може бути неточним.
                    Спробуйте зробити більш чітке фото або змініть кут зйомки.
                </p>
                <div className="warning-details">
                    <p><strong>Розпізнано:</strong> {capitalizeWords(food_type)}</p>
                    <p><strong>Точність:</strong> {Math.round(confidence * 100)}%</p>
                </div>
            </div>
        </motion.div>
    );

    const renderNutritionWarning = () => (
        <motion.div
            className="warning-container warning-nutrition-error card"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4 }}
        >
            <div className="warning-icon">ℹ️</div>
            <div className="warning-content">
                <h3>Інформація про калорії недоступна</h3>
                <p>
                    Продукт успішно розпізнано, але не вдалося отримати детальну інформацію про
                    калорійність та поживні речовини. Спробуйте пізніше або введіть дані вручну.
                </p>
                <div className="warning-details">
                    <p><strong>Розпізнано:</strong> {capitalizeWords(food_type)}</p>
                    <p><strong>Точність:</strong> {Math.round(confidence * 100)}%</p>
                </div>
            </div>
        </motion.div>
    );

    const renderSimpleResult = () => {
        const confidencePercentage = Math.round(confidence * 100);
        const confidenceColor = getConfidenceColor(confidence);

        return (
            <div className="results-content">
                <motion.div
                    className="food-identification card"
                    initial={{ scale: 0.9 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.2 }}
                >
                    <div className="food-info">
                        <h3 className="food-name">{capitalizeWords(food_type)}</h3>
                        <div className="confidence-badge">
                            <span
                                className="confidence-indicator"
                                style={{ backgroundColor: confidenceColor }}
                            ></span>
                            <span className="confidence-text">
                                {confidencePercentage}% confidence
                            </span>
                        </div>
                    </div>
                    <div className="food-emoji">
                        {getFoodEmoji(food_type)}
                    </div>
                </motion.div>

                <motion.div
                    className="nutrition-unavailable card"
                    initial={{ scale: 0.9 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.3 }}
                >
                    <div className="unavailable-content">
                        <div className="unavailable-icon">📊</div>
                        <h3>Детальна інформація про калорії недоступна</h3>
                        <p>
                            {isLowAccuracy ?
                                'Через низьку точність розпізнавання ми не можемо надати точну інформацію про поживні речовини.' :
                                'Не вдалося отримати інформацію про калорії та БЖУ для цього продукту.'
                            }
                        </p>
                        <div className="manual-entry-suggestion">
                            <p>💡 <strong>Порада:</strong> Ви можете ввести дані вручну або спробувати зробити більш чітке фото.</p>
                        </div>
                    </div>
                </motion.div>
            </div>
        );
    };
    const renderSuccessfulResult = () => {
        const { protein, fat, carbohydrates } = nutrition;
        const calories = calculateCalories(protein, fat, carbohydrates);
        const confidencePercentage = Math.round(confidence * 100);
        const confidenceColor = getConfidenceColor(confidence);

        const nutritionData = [
            {
                label: 'Protein',
                value: protein,
                color: '#FF6B6B',
                icon: '🥩'
            },
            {
                label: 'Fat',
                value: fat,
                color: '#4ECDC4',
                icon: '🥑'
            },
            {
                label: 'Carbohydrates',
                value: carbohydrates,
                color: '#45B7D1',
                icon: '🍞'
            }
        ];

        return (
            <div className="results-content">
                <motion.div
                    className="food-identification card"
                    initial={{ scale: 0.9 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.2 }}
                >
                    <div className="food-info">
                        <h3 className="food-name">{capitalizeWords(food_type)}</h3>
                        <div className="confidence-badge">
                            <span
                                className="confidence-indicator"
                                style={{ backgroundColor: confidenceColor }}
                            ></span>
                            <span className="confidence-text">
                                {confidencePercentage}% confidence
                            </span>
                        </div>
                    </div>
                    <div className="food-emoji">
                        {getFoodEmoji(food_type)}
                    </div>
                </motion.div>

                <motion.div
                    className="nutrition-overview card"
                    initial={{ scale: 0.9 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.3 }}
                >
                    <h3>Nutritional Content <span className="per-serving">(per 100g)</span></h3>

                    <div className="calories-display">
                        <div className="calories-circle">
                            <span className="calories-number">{calories}</span>
                            <span className="calories-label">kcal</span>
                        </div>
                    </div>

                    <div className="macronutrients-grid">
                        {nutritionData.map((nutrient, index) => (
                            <motion.div
                                key={nutrient.label}
                                className="nutrient-item"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.4 + index * 0.1 }}
                            >
                                <div className="nutrient-icon">{nutrient.icon}</div>
                                <div className="nutrient-info">
                                    <span className="nutrient-label">{nutrient.label}</span>
                                    <span
                                        className="nutrient-value"
                                        style={{ color: nutrient.color }}
                                    >
                                        {formatNutritionValue(nutrient.value)}
                                    </span>
                                </div>
                                <div
                                    className="nutrient-bar"
                                    style={{ backgroundColor: `${nutrient.color}20` }}
                                >
                                    <div
                                        className="nutrient-fill"
                                        style={{
                                            width: `${Math.min(nutrient.value * 2, 100)}%`,
                                            backgroundColor: nutrient.color
                                        }}
                                    ></div>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </motion.div>

                <motion.div
                    className="chart-section card"
                    initial={{ scale: 0.9 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.5 }}
                >
                    <h3>Macronutrient Breakdown</h3>
                    <NutritionChart
                        protein={protein}
                        fat={fat}
                        carbohydrates={carbohydrates}
                    />
                </motion.div>

                <motion.div
                    className="health-insights card"
                    initial={{ scale: 0.9 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.6 }}
                >
                    <h3>💡 Health Insights</h3>
                    <div className="insights-list">
                        {generateHealthInsights(protein, fat, carbohydrates, calories).map((insight, index) => (
                            <div key={index} className="insight-item">
                                <span className="insight-icon">{insight.icon}</span>
                                <span className="insight-text">{insight.text}</span>
                            </div>
                        ))}
                    </div>
                </motion.div>
            </div>
        );
    };

    return (
        <motion.div
            className="results-display"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
        >
            <div className="results-header">
                <h2>Analysis Results</h2>
                <button
                    className="btn btn-secondary new-analysis-btn"
                    onClick={onNewAnalysis}
                >
                    🔄 New Analysis
                </button>
            </div>

            <div className="results-content">

                {isLowAccuracy && renderLowAccuracyWarning()}
                {hasNutritionError && !isLowAccuracy && renderNutritionWarning()}

                {isFullySuccessful && renderSuccessfulResult()}
                {(isLowAccuracy || hasNutritionError) && !isFullySuccessful && food_type && renderSimpleResult()}
            </div>
        </motion.div>
    );
};

const getFoodEmoji = (foodType) => {
    const emojiMap = {
        'apple_pie': '🥧',
        'baby_back_ribs': '🍖',
        'baklava': '🧁',
        'beef_carpaccio': '🥩',
        'hamburger': '🍔',
        'hot_dog': '🌭',
        'pizza': '🍕',
        'sushi': '🍣',
        'tacos': '🌮',
        'ice_cream': '🍦',
    };

    return emojiMap[foodType] || '🍽️';
};

const generateHealthInsights = (protein, fat, carbs, calories) => {
    const insights = [];

    if (protein > 20) {
        insights.push({
            icon: '💪',
            text: 'High in protein - great for muscle building and repair'
        });
    }

    if (fat > 15) {
        insights.push({
            icon: '⚠️',
            text: 'High fat content - consume in moderation'
        });
    }

    if (carbs > 30) {
        insights.push({
            icon: '⚡',
            text: 'Rich in carbohydrates - good energy source'
        });
    }

    if (calories > 300) {
        insights.push({
            icon: '🔥',
            text: 'High calorie food - balance with physical activity'
        });
    }

    if (protein > 15 && fat < 10) {
        insights.push({
            icon: '✅',
            text: 'Lean protein source - excellent for fitness goals'
        });
    }

    return insights.length > 0 ? insights : [{
        icon: '📊',
        text: 'Balanced nutritional profile'
    }];
};

export default ResultsDisplay;