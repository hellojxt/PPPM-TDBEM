#pragma once
#ifndef SIMPLE_JSON_READER_H
#define SIMPLE_JSON_READER_H

#include <filesystem>
#include <cassert>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>

class SimpleJsonReader
{
public:
    SimpleJsonReader(std::filesystem::path path) {
        std::ifstream fin(path);

        std::string str, key, value;

        std::getline(fin, str);
        assert(str == "{");

        // inputDir, outputDir
        for(int i = 0; i < 2; i++)
        {
            std::getline(fin, str);
            GetKeyValue(str, key, value);
            dirMap[key] = value;
        }
        
        while(true)
        {
            std::getline(fin, str);
            if(str.find('[') != std::string::npos)
                break;
            GetKeyValue(str, key, value);
            numMap[key] = std::stof(value);
        }        

        while(true)
        {
            std::getline(fin, str);
            if(str.find(']') != std::string::npos)
                break;
            std::map<std::string, std::string> oneInfo;

            std::getline(fin, str);
            GetKeyValue(str, key, value);
            oneInfo[key] = value;

            std::getline(fin, str);
            GetKeyValue(str, key, value);
            oneInfo[key] = value;

            std::getline(fin, str);
            sceneInfoMap.emplace_back(std::move(oneInfo));
        }
        return;
    };

    std::map<std::string, std::string> dirMap;
    std::map<std::string, float> numMap;
    std::vector<std::map<std::string, std::string>> sceneInfoMap;
private:
    void GetKeyValue(std::string str, std::string& key, std::string& value)
    {
        auto delim1 = str.find(':'), delim2 = std::min(str.find(','), str.size());
        key = Trim_(str.substr(0, delim1));
        value = Trim_(str.substr(delim1 + 1, delim2 - delim1 - 1));
        return;
    };

    std::string Trim_(std::string str)
    {
        auto chIsBlankOrQuot = [](char ch) -> bool { return isblank(ch) || ch == '\"';};
        auto beginPosIt = std::find_if_not(str.begin(), str.end(), chIsBlankOrQuot);
        auto endPosIt = std::find_if_not(str.rbegin(), str.rend(), chIsBlankOrQuot);

        size_t beginPos = beginPosIt == str.end() ? 0 : beginPosIt - str.begin(),
               endPos = endPosIt == str.rbegin() ? str.size() : str.rend() - endPosIt;

        return str.substr(beginPos, endPos - beginPos);
    };
};

#endif // SIMPLE_JSON_READER_H