import React from 'react';
import { Doughnut, Bar } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    ArcElement,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import './NutritionChart.css';

ChartJS.register(
    ArcElement,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
);

const NutritionChart = ({ protein, fat, carbohydrates, chartType = 'doughnut' }) => {
    const data = {
        labels: ['Protein', 'Fat', 'Carbohydrates'],
        datasets: [
            {
                data: [protein, fat, carbohydrates],
                backgroundColor: [
                    '#FF6B6B',
                    '#4ECDC4',
                    '#45B7D1',
                ],
                borderColor: [
                    '#FF5252',
                    '#26C6DA',
                    '#2196F3',
                ],
                borderWidth: 2,
                hoverBackgroundColor: [
                    '#FF5252',
                    '#26C6DA',
                    '#2196F3',
                ],
                hoverBorderWidth: 3,
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    padding: 20,
                    usePointStyle: true,
                    font: {
                        size: 12,
                    },
                },
            },
            tooltip: {
                callbacks: {
                    label: function (context) {
                        const label = context.label || '';
                        const value = context.parsed || context.raw;
                        return `${label}: ${value.toFixed(1)}g`;
                    },
                },
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                titleColor: 'white',
                bodyColor: 'white',
                cornerRadius: 8,
            },
        },
        cutout: chartType === 'doughnut' ? '60%' : 0,
    };

    const barOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false,
            },
            tooltip: {
                callbacks: {
                    label: function (context) {
                        return `${context.parsed.y.toFixed(1)}g`;
                    },
                },
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                titleColor: 'white',
                bodyColor: 'white',
                cornerRadius: 8,
            },
        },
        scales: {
            y: {
                beginAtZero: true,
                ticks: {
                    callback: function (value) {
                        return value + 'g';
                    },
                },
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)',
                },
            },
            x: {
                grid: {
                    display: false,
                },
            },
        },
    };

    return (
        <div className="nutrition-chart">
            <div className="chart-container">
                {chartType === 'doughnut' ? (
                    <Doughnut data={data} options={options} />
                ) : (
                    <Bar data={data} options={barOptions} />
                )}
            </div>

            <div className="chart-summary">
                <div className="total-macros">
                    <span className="total-label">Total Macros:</span>
                    <span className="total-value">
                        {(protein + fat + carbohydrates).toFixed(1)}g
                    </span>
                </div>

                <div className="macro-percentages">
                    {['Protein', 'Fat', 'Carbohydrates'].map((macro, index) => {
                        const values = [protein, fat, carbohydrates];
                        const total = values.reduce((sum, val) => sum + val, 0);
                        const percentage = ((values[index] / total) * 100).toFixed(1);

                        return (
                            <div key={macro} className="macro-percentage">
                                <span
                                    className="percentage-indicator"
                                    style={{ backgroundColor: data.datasets[0].backgroundColor[index] }}
                                ></span>
                                <span className="percentage-text">
                                    {macro}: {percentage}%
                                </span>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};

export default NutritionChart;