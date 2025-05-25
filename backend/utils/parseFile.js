export const extractJson = (text) => {
    const match = text.match(/{[\s\S]*}/);
    return match ? JSON.parse(match[0]) : null;
};